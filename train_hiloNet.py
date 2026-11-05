"""Train the HiLoNet SMR models for WiFi, RFID and mmWave from one script.

Each modality trains its own independent model, one after another. Every run is
re-seeded, so a modality gets the same result under `--modality all` as it does
when trained alone. Checkpoints are saved under ./result/params/<split>/ and evaluated
with eval_hiloNet.py.

Usage:
    python train_hiloNet.py                              # all three modalities
    python train_hiloNet.py --modality wifi --epoch 50
    python train_hiloNet.py --modality rfid mmwave --gpu 1
"""

import argparse
import datetime
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_GPU_REMAPPED = False
if __name__ == '__main__':
    # Some environments have a missing/unwritable /tmp, which breaks DataLoader workers.
    _project_tmp_dir = os.path.join('.', 'result', 'tmp')
    os.makedirs(_project_tmp_dir, exist_ok=True)
    for _tmp_var in ('TMPDIR', 'TEMP', 'TMP'):
        os.environ.setdefault(_tmp_var, _project_tmp_dir)

    # Must happen before CUDA is initialized. --gpu then maps to local cuda:0.
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        _gpu_parser = argparse.ArgumentParser(add_help=False)
        _gpu_parser.add_argument('--gpu', type=int, default=0)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(_gpu_parser.parse_known_args()[0].gpu)
        _GPU_REMAPPED = True

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from tqdm import tqdm

import XRFDataset


NUM_CLASSES = 55
MASK_BALANCE_LAMBDA = 1e-2
TRAIN_SCENE = 'scene1'
NORM_EPS = 1e-6
TRANSFORMER_DIM = 128
TRANSFORMER_HEADS = 8
TRANSFORMER_DROPOUT = 0.1

MODALITIES = {
    'wifi': {'label': 'WiFi', 'batch_index': 0, 'model_name': 'wifi_hilonet_SMR'},
    'rfid': {'label': 'RFID', 'batch_index': 1, 'model_name': 'rfid_hilonet_SMR'},
    'mmwave': {'label': 'mmWave', 'batch_index': 2, 'model_name': 'mmwave_hilonet_SMR'},
}


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Needed for deterministic cuBLAS (CUDA >= 10.2)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(args):
    if not torch.cuda.is_available():
        print("[WARN] CUDA is not available: training on CPU.")
        return torch.device('cpu')
    gpu = 0 if _GPU_REMAPPED else args.gpu
    if gpu < 0 or gpu >= torch.cuda.device_count():
        raise ValueError(f"Invalid --gpu index {gpu}. Available: 0-{torch.cuda.device_count() - 1}")
    torch.cuda.set_device(gpu)
    print(f"[GPU] Using cuda:{gpu} ({torch.cuda.get_device_name(gpu)})")
    return torch.device(f'cuda:{gpu}')


def fft_transform(x):
    """Normalized log-magnitude rFFT of [B, C, T], returned as [B, C, F]."""
    x = x - x.mean(dim=-1, keepdim=True)
    fft_x = torch.log1p(torch.abs(torch.fft.rfft(x, dim=-1)))
    fft_mean = fft_x.mean(dim=-1, keepdim=True)
    fft_std = fft_x.std(dim=-1, keepdim=True)
    return (fft_x - fft_mean) / (fft_std + NORM_EPS)


def align_mmwave_to_bthw(x):
    """Align a raw mmWave tensor to [B, T, H, W]."""
    if x.dim() == 5 and x.size(1) == 1:
        x = x.squeeze(1)
    if x.dim() != 4:
        raise ValueError(f"Expected 4D/5D mmWave, got {tuple(x.shape)}")
    if x.size(1) > 256 and x.size(3) <= 256:
        return x.permute(0, 3, 1, 2).contiguous()
    return x.contiguous()


class HiLoNetBase(nn.Module):
    """Soft spectral mask and class-wise fusion shared by all three modalities.

    Subclass forward() returns:
        (fused_log_probs, logits_cnn, logits_transformer, stream_weights, mask)
    """

    def __init__(self, num_classes, init_stream_logits):
        super().__init__()
        self.num_classes = int(num_classes)
        self.stream_weight_logits = nn.Parameter(init_stream_logits)
        self.class_weight_delta = nn.Parameter(torch.zeros(2, self.num_classes))
        self.mask_logits = nn.Parameter(torch.zeros(1, 1, 1))  # Resized on first forward

    def _spectral_mask(self, x, freq_bins):
        """Return M(f) in [0, 1] as [1, 1, F]."""
        if self.mask_logits.shape[-1] != freq_bins:
            self.mask_logits = nn.Parameter(torch.zeros(1, 1, freq_bins, device=x.device, dtype=x.dtype))
        return torch.sigmoid(self.mask_logits)

    def _fuse(self, logits_cnn, logits_transformer):
        probs_cnn = torch.softmax(logits_cnn, dim=1)
        probs_transformer = torch.softmax(logits_transformer, dim=1)
        class_weight_logits = self.stream_weight_logits.unsqueeze(1) + self.class_weight_delta
        stream_weights = torch.softmax(class_weight_logits, dim=0)
        fused_probs = stream_weights[0].unsqueeze(0) * probs_cnn + stream_weights[1].unsqueeze(0) * probs_transformer
        return torch.log(fused_probs.clamp_min(1e-8)), stream_weights


def _equal_stream_logits():
    return torch.log(torch.tensor([0.5, 0.5], dtype=torch.float32))


def _build_transformer(layers, ffn):
    enc_layer = nn.TransformerEncoderLayer(
        d_model=TRANSFORMER_DIM,
        nhead=TRANSFORMER_HEADS,
        dim_feedforward=ffn,
        dropout=TRANSFORMER_DROPOUT,
        activation='gelu',
        batch_first=True,
    )
    return nn.TransformerEncoder(enc_layer, num_layers=layers)


class HiLoNet60(HiLoNetBase):
    """WiFi HiLoNet over [B, 270, T]: (1 - mask) goes to the CNN, mask to the Transformer."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__(num_classes, _equal_stream_logits())

        self.cnn = nn.Sequential(
            nn.Conv1d(270, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc_cnn = nn.Linear(256, num_classes)

        self.trans_in_proj = nn.Conv1d(270, TRANSFORMER_DIM, kernel_size=1)
        self.transformer = _build_transformer(layers=4, ffn=512)
        self.trans_norm = nn.LayerNorm(TRANSFORMER_DIM)
        self.fc_transformer = nn.Linear(TRANSFORMER_DIM, num_classes)

    def forward(self, x):
        x_fft = fft_transform(x)
        freq_bins = x_fft.shape[-1]
        mask = self._spectral_mask(x_fft, freq_bins).expand(x_fft.size(0), x_fft.size(1), freq_bins)

        logits_cnn = self.fc_cnn(self.cnn((1.0 - mask) * x_fft).view(x.size(0), -1))

        z = self.trans_in_proj(mask * x_fft).transpose(1, 2).contiguous()
        z = self.trans_norm(self.transformer(z)).mean(dim=1)
        logits_transformer = self.fc_transformer(z)

        fused_log_probs, stream_weights = self._fuse(logits_cnn, logits_transformer)
        return fused_log_probs, logits_cnn, logits_transformer, stream_weights, mask


class HiLoNet60_RFID(HiLoNetBase):
    """RFID HiLoNet over [B, 23, T]: (1 - mask) goes to the CNN, mask to the Transformer."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__(num_classes, _equal_stream_logits())

        self.cnn_layers = nn.ModuleList([
            nn.Conv1d(23, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        ])
        self.cnn_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_cnn = nn.Linear(256, num_classes)

        self.trans_in_proj = nn.Conv1d(23, TRANSFORMER_DIM, kernel_size=1)
        self.transformer = _build_transformer(layers=2, ffn=256)
        self.trans_norm = nn.LayerNorm(TRANSFORMER_DIM)
        self.fc_transformer = nn.Linear(TRANSFORMER_DIM, num_classes)

    def forward(self, x):
        x_fft = fft_transform(x)
        freq_bins = x_fft.shape[-1]
        mask = self._spectral_mask(x_fft, freq_bins).expand(x_fft.size(0), x_fft.size(1), freq_bins)

        y = (1.0 - mask) * x_fft
        for layer in self.cnn_layers:
            y = layer(y)
        logits_cnn = self.fc_cnn(self.cnn_pool(y).view(x.size(0), -1))

        z = self.trans_in_proj(mask * x_fft).transpose(1, 2).contiguous()
        z = self.trans_norm(self.transformer(z)).mean(dim=1)
        logits_transformer = self.fc_transformer(z)

        fused_log_probs, stream_weights = self._fuse(logits_cnn, logits_transformer)
        return fused_log_probs, logits_cnn, logits_transformer, stream_weights, mask


class SimpleCNN2D_mmWave(nn.Module):
    """2D CNN for mmWave [B, T, H, W], treating T as input channels."""

    def __init__(self, num_classes=NUM_CLASSES, time_frames=10):
        super().__init__()
        self.conv1 = nn.Conv2d(time_frames, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        return self.fc(x.view(x.size(0), -1))


class TemporalTransformer_mmWave(nn.Module):
    """Temporal Transformer for mmWave [B, T, H, W]."""

    def __init__(self, num_classes=NUM_CLASSES, embed_dim=TRANSFORMER_DIM, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = None  # Created on first forward, once H*W is known
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, T, H, W = x.shape
        if self.proj is None:
            self.proj = nn.Linear(H * W, self.embed_dim).to(x.device)
        x = self.proj(x.reshape(B, T, H * W))
        return self.fc(self.transformer(x).mean(dim=1))


class HiLoNet60_mmWave(HiLoNetBase):
    """mmWave HiLoNet: FFT along H, then (1 - mask) to the CNN and mask to the Transformer."""

    def __init__(self, num_classes=NUM_CLASSES, time_frames=10):
        super().__init__(num_classes, torch.zeros(2))
        self.cnn_branch = SimpleCNN2D_mmWave(num_classes=num_classes, time_frames=time_frames)
        self.transformer_branch = TemporalTransformer_mmWave(num_classes=num_classes)

    def forward(self, x):
        x = align_mmwave_to_bthw(x)
        x = torch.log1p(torch.fft.rfft(x, dim=2).abs())

        freq_bins = x.shape[2]
        mask = self._spectral_mask(x, freq_bins).expand(x.shape[0], x.shape[1], freq_bins).unsqueeze(-1)

        logits_cnn = self.cnn_branch((1.0 - mask) * x)
        logits_transformer = self.transformer_branch(mask * x)

        fused_log_probs, stream_weights = self._fuse(logits_cnn, logits_transformer)
        return fused_log_probs, logits_cnn, logits_transformer, stream_weights, mask


def build_model(modality, sample):
    """Build the model for a modality. `sample` is one batch of that modality's input."""
    if modality == 'mmwave':
        return HiLoNet60_mmWave(time_frames=align_mmwave_to_bthw(sample).shape[1])
    return HiLoNet60() if modality == 'wifi' else HiLoNet60_RFID()


def materialize_lazy_parameters(model, sample):
    """Create input-size-dependent parameters (mask_logits, mmWave proj) with one forward.

    Call this before building the optimizer: parameters created later are never
    updated, and because zero_grad() skips them, their gradients accumulate and
    distort clip_grad_norm_. Eval mode leaves the BatchNorm statistics untouched.
    """
    model.eval()
    with torch.no_grad():
        model(sample)


def run_epoch(model, loader, batch_index, device, optimizer=None, desc=""):
    """Run one epoch over `loader`, training if an optimizer is given. Returns (mean loss, accuracy %)."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct, total = 0, 0

    with torch.set_grad_enabled(is_train):
        for batch in tqdm(loader, desc=desc, leave=False):
            inputs = batch[batch_index].to(device).float()
            labels = batch[3].to(device)

            fused_log_probs, logits_cnn, logits_transformer, _, mask = model(inputs)
            loss_fused = F.nll_loss(fused_log_probs, labels)
            loss_cnn = F.cross_entropy(logits_cnn, labels)
            loss_transformer = F.cross_entropy(logits_transformer, labels)
            loss = (loss_fused + loss_cnn + loss_transformer) / 3.0

            if is_train:
                # Encourage a balanced allocation of spectrum between the two streams.
                loss = loss + MASK_BALANCE_LAMBDA * (mask.mean() - 0.5) ** 2
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            correct += (fused_log_probs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / max(len(loader), 1), 100.0 * correct / max(total, 1)


def train_modality(modality, base_args, device):
    """Train one modality from scratch. Returns its best validation accuracy."""
    config = MODALITIES[modality]
    label = config['label']
    args = argparse.Namespace(**dict(vars(base_args), modality=modality))
    start_time = datetime.datetime.now()

    seed_everything(args.seed)

    split_name = XRFDataset.get_split_name()
    model_dir = os.path.join('./result/params', split_name, config['model_name'])
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, f"{config['model_name']}_best.pth")
    final_model_path = os.path.join(model_dir, f"{config['model_name']}_final.pth")

    train_dataset = XRFDataset.XRFBertDatasetNewMix(is_train=True, scene=TRAIN_SCENE)
    val_dataset = XRFDataset.XRFBertDatasetNewMix(is_train=False, scene=TRAIN_SCENE, data_split='val_data')

    def make_loader(dataset, shuffle, seed):
        return Data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            pin_memory=torch.cuda.is_available(),
            num_workers=args.num_workers,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(seed),
        )

    train_loader = make_loader(train_dataset, True, args.seed)
    val_loader = make_loader(val_dataset, False, args.seed + 1)

    try:
        sample = next(iter(train_loader))[config['batch_index']]
    except StopIteration:
        raise RuntimeError(f"Empty train loader for {label}: cannot infer the input shape.")

    model = build_model(modality, sample).to(device)
    materialize_lazy_parameters(model, sample.to(device).float())
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"\n{label} ({split_name}): {len(train_dataset)} train, {len(val_dataset)} val samples, "
        f"{num_params:,} parameters"
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=args.lr * 0.01)

    train_losses, val_losses = [], []
    best_val_acc = -1.0

    def full_checkpoint(epoch):
        return {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_acc': best_val_acc,
            'args': vars(args),
        }

    for epoch in range(args.epoch):
        train_loss, train_acc = run_epoch(
            model, train_loader, config['batch_index'], device, optimizer=optimizer, desc="Training"
        )
        train_losses.append(train_loss)

        val_loss, val_acc = run_epoch(model, val_loader, config['batch_index'], device, desc="Validation")
        val_losses.append(val_loss)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save(full_checkpoint(epoch), best_model_path)
        print(
            f"[{label}] Epoch {epoch + 1}/{args.epoch} | train loss {train_loss:.4f} acc {train_acc:.2f}% "
            f"| val loss {val_loss:.4f} acc {val_acc:.2f}%{' *best' if is_best else ''}"
        )

        scheduler.step()

    torch.save(full_checkpoint(args.epoch - 1), final_model_path)

    duration = (datetime.datetime.now() - start_time).seconds
    print(f"{label} done in {duration // 60}m {duration % 60}s, best val accuracy {best_val_acc:.2f}%. Checkpoints: {model_dir}")
    return best_val_acc


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Train HiLoNet SMR models for WiFi, RFID and mmWave')
    parser.add_argument('--modality', nargs='+', default=['all'], choices=['all'] + list(MODALITIES),
                        help='Modalities to train, in order (default: all)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for AdamW')
    parser.add_argument('--epoch', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args(argv)


def main(args):
    modalities = list(MODALITIES) if 'all' in args.modality else list(dict.fromkeys(args.modality))
    device = resolve_device(args)

    for modality in modalities:
        train_modality(modality, args, device)


if __name__ == '__main__':
    main(parse_args())
