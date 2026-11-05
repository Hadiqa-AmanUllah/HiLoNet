"""Evaluate the HiLoNet SMR models trained by train_hiloNet.py.

By default, evaluates each model's best checkpoint from the active split on every scene:
scene 1 val_data (also used to select that checkpoint), scene 1 test_data (split mode 0
only) and the test_data of scenes 2, 3 and 4. Prints a results table for the individual
models, then one for soft voting (averaged class probabilities) over every combination of
the evaluated modalities. Results are saved per split, model and scene under ./result/eval/.

Usage:
    python eval_hiloNet.py                       # all three modalities, all scenes
    python eval_hiloNet.py --modality wifi
    python eval_hiloNet.py --modality rfid mmwave --scene 2 --gpu 1
    python eval_hiloNet.py --scene val           # scene 1 val_data only
"""

import argparse
import itertools
import os

if __name__ == '__main__':
    # Some environments have a missing/unwritable /tmp, which breaks DataLoader workers.
    _project_tmp_dir = os.path.join('.', 'result', 'tmp')
    os.makedirs(_project_tmp_dir, exist_ok=True)
    for _tmp_var in ('TMPDIR', 'TEMP', 'TMP'):
        os.environ.setdefault(_tmp_var, _project_tmp_dir)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.utils.data as Data
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm

import XRFDataset
import train_hiloNet
from train_hiloNet import MODALITIES, NUM_CLASSES, build_model, materialize_lazy_parameters


SCENES = ['1', '2', '3', '4']
SCENE1_VAL = ('scene1', 'val_data', 'scene1_val.txt')


def test_target(scene_number):
    return f'scene{scene_number}', 'test_data', f'scene{scene_number}_test.txt'


def has_labels(label_file):
    path = os.path.join(XRFDataset._resolve_dataset_root(), label_file)
    return os.path.isfile(path) and os.path.getsize(path) > 0


def eval_targets(scene_arg):
    """Return the (scene, data_split, label_file) sets selected by --scene."""
    if scene_arg == 'val':
        return [SCENE1_VAL]
    if scene_arg != 'all':
        return [test_target(scene_arg)]
    targets = []
    for target in [SCENE1_VAL] + [test_target(number) for number in SCENES]:
        if has_labels(target[2]):
            targets.append(target)
        else:
            print(f"[Skip] {target[0]} {target[1]}: no samples in the active split")
    if not targets:
        raise RuntimeError("No label files found. Run 'python split_train_test.py' first.")
    return targets


def make_loader(target, args):
    scene, split, label_file = target
    dataset = XRFDataset.XRFBertDatasetNewMix(is_train=False, scene=scene, data_split=split, label_file=label_file)
    return Data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=args.num_workers,
        drop_last=False,
    )


def load_model(modality, checkpoint, sample, device):
    """Rebuild a model and load its weights from a checkpoint."""
    model = build_model(modality, sample).to(device)
    materialize_lazy_parameters(model, sample[:1].to(device).float())
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def predict(model, loader, batch_index, device, desc):
    """Return (y_true, class probabilities) as numpy arrays."""
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            fused_log_probs = model(batch[batch_index].to(device).float())[0]
            all_probs.append(fused_log_probs.exp().cpu().numpy())
            all_labels.append(batch[3].numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


def save_confusion_matrix(cm, label, path):
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'{label} HiLoNet-SMR Confusion Matrix ({NUM_CLASSES} classes)', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_summary(results, label, path):
    with open(path, 'w') as f:
        f.write(f"{label} HiLoNet-SMR Evaluation Results\n")
        f.write("=" * 60 + "\n")
        for name, value in results.items():
            if isinstance(value, float):
                f.write(f"{name:<16}: {value:.4f}\n")
            else:
                f.write(f"{name:<16}: {value}\n")


def score_and_save(name, label, target, y_true, probs, extra=None):
    """Compute metrics for one model on one target, save them and return them."""
    scene, split, _ = target
    y_pred = probs.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    results = {
        'scene': scene,
        'split': split,
        'accuracy': float((y_true == y_pred).mean()),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'num_samples': len(y_true),
    }
    results.update(extra or {})

    output_dir = os.path.join('./result/eval', XRFDataset.get_split_name(), name, f'{scene}_{split}')
    os.makedirs(output_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    save_confusion_matrix(cm, label, os.path.join(output_dir, f'{name}_confusion_matrix.png'))
    save_summary(results, label, os.path.join(output_dir, f'eval_{name}_results.txt'))
    return results


def evaluate_modality(modality, targets, args, device):
    """Evaluate one modality's best checkpoint. Returns (results, {target: (y_true, probs)})."""
    config = MODALITIES[modality]
    label = config['label']
    model_name = config['model_name']
    model_path = os.path.join('./result/params', XRFDataset.get_split_name(), model_name, f'{model_name}_best.pth')

    loaders = [make_loader(target, args) for target in targets]
    sample = next(iter(loaders[0]))[config['batch_index']]
    checkpoint = torch.load(model_path, map_location=device)
    model = load_model(modality, checkpoint, sample, device)
    mask = torch.sigmoid(model.mask_logits).detach().cpu()
    print(f"{label} soft mask: mean={mask.mean().item():.4f} min={mask.min().item():.4f} max={mask.max().item():.4f}")
    extra = {
        'mask_mean': mask.mean().item(),
        'mask_min': mask.min().item(),
        'mask_max': mask.max().item(),
        'model_path': model_path,
    }

    all_results, predictions = [], {}
    for target, loader in zip(targets, loaders):
        y_true, probs = predict(model, loader, config['batch_index'], device, f"{label} {target[0]} {target[1]}")
        predictions[target] = (y_true, probs)
        all_results.append(score_and_save(model_name, label, target, y_true, probs, extra))
    return all_results, predictions


def evaluate_ensemble(combo, targets, predictions):
    """Soft voting: average the class probabilities of the modalities in combo."""
    label = '+'.join(MODALITIES[modality]['label'] for modality in combo)
    name = 'ensemble_' + '_'.join(combo)
    all_results = []
    for target in targets:
        y_true = predictions[combo[0]][target][0]
        if not all(np.array_equal(y_true, predictions[modality][target][0]) for modality in combo):
            raise RuntimeError(f"{label}: modalities disagree on the labels of {target[0]} {target[1]}.")
        probs = np.mean([predictions[modality][target][1] for modality in combo], axis=0)
        all_results.append(score_and_save(name, label, target, y_true, probs))
    return label, all_results


def print_table(title, rows):
    """Print (name, results list) rows as a table of metrics."""
    print(f"\n{title}")
    print(f"{'Model':<19}{'Scene':<8}{'Data':<12}{'Samples':>8}{'Accuracy':>10}{'Precision':>11}{'Recall':>8}{'F1':>8}")
    print("-" * 84)
    for name, all_results in rows:
        for r in all_results:
            print(
                f"{name:<19}{r['scene']:<8}{r['split']:<12}{r['num_samples']:>8}"
                f"{r['accuracy']:>10.4f}{r['precision']:>11.4f}{r['recall']:>8.4f}{r['f1']:>8.4f}"
            )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate HiLoNet SMR models for WiFi, RFID and mmWave')
    parser.add_argument('--modality', nargs='+', default=['all'], choices=['all'] + list(MODALITIES),
                        help='Modalities to evaluate, in order (default: all)')
    parser.add_argument('--scene', default='all', choices=['all', 'val'] + SCENES,
                        help='all: every scene (default). val: scene 1 val_data. '
                             '1-4: that scene\'s test_data (scene 1 needs split mode 0).')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of dataloader workers')
    return parser.parse_args(argv)


def main(args):
    modalities = list(MODALITIES) if 'all' in args.modality else list(dict.fromkeys(args.modality))
    device = train_hiloNet.resolve_device(args)

    targets = eval_targets(args.scene)
    rows, predictions = [], {}
    for modality in modalities:
        all_results, predictions[modality] = evaluate_modality(modality, targets, args, device)
        rows.append((MODALITIES[modality]['label'], all_results))
    print_table('Individual models', rows)

    combos = [combo for size in range(2, len(modalities) + 1) for combo in itertools.combinations(modalities, size)]
    if combos:
        print_table('Soft voting ensembles', [evaluate_ensemble(combo, targets, predictions) for combo in combos])

    print(f"\nSaved to: {os.path.join('./result/eval', XRFDataset.get_split_name())}")


if __name__ == '__main__':
    main(parse_args())
