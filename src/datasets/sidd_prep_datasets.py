"""
SIDD sRGB datasets for pre-cropped PNG patches.

Training data layout:
    data_dir/
        RN/  (real noisy)
            0_0_0.png, 0_0_1.png, ...
        CL/  (clean / GT)
            0_0_0.png, 0_0_1.png, ...

Validation/Test data layout:
    data_dir/
        Noisy/
            31_09.PNG, ...
        GT/
            31_09.PNG, ...

All images are loaded as float32 in [0, 255] range (no /255 normalization).
"""

import os
import glob
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def _aug_np(img, flip_h, flip_w, transpose):
    """Data augmentation for (C, H, W) numpy array."""
    if flip_h:
        img = img[:, ::-1, :]
    if flip_w:
        img = img[:, :, ::-1]
    if transpose:
        img = np.transpose(img, (0, 2, 1))
    return img


def _crop_np(img, patch_size, ph, pw):
    return img[:, ph:ph+patch_size, pw:pw+patch_size]


class SIDDPrepTrainDataset(Dataset):
    """Pre-cropped SIDD sRGB training patches (RN/ + CL/ folders)."""

    def __init__(self, dataset_opt):
        super().__init__()
        self.data_dir = dataset_opt.data_dir
        self.patch_size = dataset_opt.patch_size

        rn_dir = os.path.join(self.data_dir, 'RN')
        cl_dir = os.path.join(self.data_dir, 'CL')

        rn_paths = sorted(glob.glob(os.path.join(rn_dir, '*.png')))
        cl_paths = sorted(glob.glob(os.path.join(cl_dir, '*.png')))
        assert len(rn_paths) == len(cl_paths), \
            f"RN ({len(rn_paths)}) and CL ({len(cl_paths)}) count mismatch"

        self.pairs = list(zip(rn_paths, cl_paths))
        print(f'[SIDDPrepTrainDataset] {len(self.pairs)} training pairs loaded')

    def _load(self, path):
        img = np.array(Image.open(path).convert('RGB'), dtype=np.float32)
        return np.transpose(img, (2, 0, 1))  # (C, H, W), [0, 255]

    def __getitem__(self, index):
        index = index % len(self.pairs)
        rn_path, cl_path = self.pairs[index]

        noisy = self._load(rn_path)
        clean = self._load(cl_path)

        C, H, W = noisy.shape
        if H > self.patch_size and W > self.patch_size:
            ph = random.randint(0, H - self.patch_size)
            pw = random.randint(0, W - self.patch_size)
            noisy = _crop_np(noisy, self.patch_size, ph, pw)
            clean = _crop_np(clean, self.patch_size, ph, pw)

        flip_h = random.random() > 0.5
        flip_w = random.random() > 0.5
        transpose = random.random() > 0.5
        noisy = _aug_np(noisy, flip_h, flip_w, transpose)
        clean = _aug_np(clean, flip_h, flip_w, transpose)

        noisy = np.ascontiguousarray(noisy)
        clean = np.ascontiguousarray(clean)
        return 'None', {'noisy': noisy, 'gt': clean}

    def __len__(self):
        return 100000


class SIDDPrepValidationDataset(Dataset):
    """SIDD sRGB validation/test PNG blocks (Noisy/ + GT/ folders)."""

    def __init__(self, dataset_opt):
        super().__init__()
        self.data_dir = dataset_opt.data_dir

        noisy_dir = os.path.join(self.data_dir, 'Noisy')
        gt_dir = os.path.join(self.data_dir, 'GT')

        noisy_paths = sorted(glob.glob(os.path.join(noisy_dir, '*.[pP][nN][gG]')))
        gt_paths = sorted(glob.glob(os.path.join(gt_dir, '*.[pP][nN][gG]')))
        assert len(noisy_paths) == len(gt_paths), \
            f"Noisy ({len(noisy_paths)}) and GT ({len(gt_paths)}) count mismatch"

        self.pairs = list(zip(noisy_paths, gt_paths))
        print(f'[SIDDPrepValidationDataset] {len(self.pairs)} validation pairs loaded')

    def _load(self, path):
        img = np.array(Image.open(path).convert('RGB'), dtype=np.float32)
        return np.transpose(img, (2, 0, 1))  # (C, H, W), [0, 255]

    def __getitem__(self, index):
        noisy_path, gt_path = self.pairs[index]
        name = os.path.splitext(os.path.basename(noisy_path))[0]

        noisy = self._load(noisy_path)
        gt = self._load(gt_path)
        return name, {'noisy': noisy, 'gt': gt}

    def __len__(self):
        return len(self.pairs)
