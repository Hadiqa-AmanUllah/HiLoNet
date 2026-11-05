import logging
import torch
import os
from torch.utils.data.dataset import Dataset
import numpy as np

from split_train_test import SPLIT_OUTPUT_ROOT

log = logging.getLogger(__name__)

_MODALITY_SHAPES = {
    'WiFi': (270, 1000),
    'RFID': (23, 1000),
    'mmWave': (17, 256, 128),
}

def _resolve_dataset_root():
    """Resolve dataset root to a relative path (robust to directory renaming)."""
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), SPLIT_OUTPUT_ROOT))
    return base + os.sep


def get_split_name():
    """Return the active split's name ('split0'...), which separates each split's results."""
    path = os.path.join(_resolve_dataset_root(), 'split_name.txt')
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[XRFDataset] {path} not found. Run 'python split_train_test.py' first.")
    with open(path) as f:
        return f.read().strip()


def _scene_folder(scene):
    """Normalize a scene identifier to its 'sceneN' directory/prefix name."""
    scene = str(scene)
    return scene if scene.lower().startswith('scene') else f"scene{scene}"


def _modality_paths(root, *nested):
    """Build (wifi, rfid, mmwave) subfolder paths under root/*nested."""
    return tuple(os.path.join(root, *nested, name) for name in _MODALITY_SHAPES)


class XRFBertDatasetNewMix(Dataset):
    def __init__(
        self,
        is_train=True,
        scene='dml',
        label_file=None,
        data_split=None,
    ):
        super(XRFBertDatasetNewMix, self).__init__()
        self.word_list = None
        self.file_path = _resolve_dataset_root()
        self.is_train = is_train
        self.scene = scene
        self.data_split = data_split if data_split is not None else ('train_data' if self.is_train else 'val_data')

        label_root = self.file_path
        if label_file is not None:
            self.file = os.path.join(label_root, label_file)
        else:
            split_suffix = 'train' if self.is_train else 'val'
            self.file = os.path.join(label_root, f"{_scene_folder(self.scene)}_{split_suffix}.txt")

        self.data = {
            'file_name': [],
            'label': []
        }

        self._set_data_paths()

        if not os.path.isfile(self.file):
            raise FileNotFoundError(
                f"\n[XRFDataset] Label file not found: {self.file}\n"
                f"  Dataset root resolved to: {self.file_path}\n"
                f"  Run 'python split_train_test.py' first to generate label files."
            )
        with open(self.file) as file:
            val_list = file.readlines()
        if not val_list:
            raise ValueError(f"[XRFDataset] Label file {self.file} is empty. Please check label generation.")
        for string in val_list:
            parts = string.strip().split(',')
            if len(parts) < 3:
                raise ValueError(f"[XRFDataset] Malformed line in label file {self.file}: {string}")
            self.data['file_name'].append(parts[0])
            self.data['label'].append(int(parts[2]) - 1)
        log.info("load XRF dataset")

    def _set_data_paths(self):
        """Set path_wifi/path_rfid/path_mmwave under file_path/{split}/{scene}/{modality}/."""
        self.path = self.file_path
        split_folder = self.data_split if self.data_split is not None else ('train_data' if self.is_train else 'val_data')
        scene_folder = _scene_folder(self.scene)
        self.path_wifi, self.path_rfid, self.path_mmwave = _modality_paths(self.file_path, split_folder, scene_folder)

    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, idx):
        file_name = self.data['file_name'][idx]
        label = self.data['label'][idx]
        # Remove leading scene folder if present in file_name
        scene_prefix = _scene_folder(self.scene) + "/"
        file_name_mod = file_name[len(scene_prefix):] if file_name.startswith(scene_prefix) else file_name

        wifi_data = load_modality(file_name_mod, 'WiFi', self.path_wifi)
        rfid_data = load_modality(file_name_mod, 'RFID', self.path_rfid)
        mmwave_data = load_modality(file_name_mod, 'mmWave', self.path_mmwave)

        if rfid_data.dim() == 2:
            if rfid_data.shape[0] == 1 and rfid_data.shape[1] >= 23:
                rfid_data = rfid_data.expand(23, rfid_data.shape[1])
            elif rfid_data.shape[0] != 23:
                if rfid_data.shape[0] < 23:
                    pad = torch.zeros(23 - rfid_data.shape[0], rfid_data.shape[1])
                    rfid_data = torch.cat([rfid_data, pad], dim=0)
                else:
                    rfid_data = rfid_data[:23, :]
        elif rfid_data.dim() == 1:
            rfid_data = rfid_data.unsqueeze(0).expand(23, rfid_data.shape[0])

        if mmwave_data.dim() == 1:
            if mmwave_data.numel() == 17 * 256 * 128:
                mmwave_data = mmwave_data.view(17, 256, 128)
            else:
                mmwave_data = torch.zeros(17, 256, 128).float()
        elif mmwave_data.dim() == 2:
            if mmwave_data.shape[0] == 1 and mmwave_data.shape[1] == 17 * 256 * 128:
                mmwave_data = mmwave_data.view(17, 256, 128)
            elif mmwave_data.shape[0] == 17 and mmwave_data.shape[1] == 256 * 128:
                mmwave_data = mmwave_data.view(17, 256, 128)
            else:
                mmwave_data = torch.zeros(17, 256, 128).float()
        elif mmwave_data.dim() == 3:
            if mmwave_data.shape != (17, 256, 128):
                mmwave_data = torch.zeros(17, 256, 128).float()
        else:
            mmwave_data = torch.zeros(17, 256, 128).float()

        dummy_vec = torch.zeros(768)
        return wifi_data, rfid_data, mmwave_data, label, dummy_vec


def _load_npy(filename, path, shape, squeeze_batch=False):
    """Load a modality .npy file, using a zero tensor of `shape` if it is missing or corrupt."""
    filepath = os.path.join(path, filename + '.npy')
    if not os.path.exists(filepath):
        log.warning("Missing data file, using zeros: %s", filepath)
        return torch.zeros(*shape).float()
    try:
        data = np.load(filepath)
    except (ValueError, EOFError, OSError):
        log.warning("Corrupt data file, using zeros: %s", filepath)
        return torch.zeros(*shape).float()
    if squeeze_batch and data.ndim == 4 and data.shape[0] == 1:
        data = data[0]
    return torch.from_numpy(data).float()


def load_modality(filename, modality, path):
    """Load one modality's processed .npy file (path already points at its folder)."""
    return _load_npy(filename, path, _MODALITY_SHAPES[modality], squeeze_batch=(modality == 'mmWave'))
