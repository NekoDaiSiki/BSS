import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class ImagenetValDataset(Dataset):

    def __init__(self, dataset_opt):
        super(ImagenetValDataset, self).__init__()
        self.data_dir = dataset_opt.data_dir
        self.patch = dataset_opt.patch
        self.train_fns = glob.glob(os.path.join(self.data_dir, '*'))
        self.train_fns.sort()
        print('[NOTE]: fetch {} samples for training'.format(len(self.train_fns)))

    def __getitem__(self, index):
        fn = self.train_fns[index]
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        H = im.shape[0]
        W = im.shape[1]
        if H - self.patch > 0:
            xx = np.random.randint(0, H - self.patch)
            im = im[xx:xx + self.patch, :, :]
        if W - self.patch > 0:
            yy = np.random.randint(0, W - self.patch)
            im = im[:, yy:yy + self.patch, :]
        im = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
        return im

    def __len__(self):
        return len(self.train_fns)

class BasicValidateDataset(Dataset):

    def __init__(self, dataset_opt):
        self.data_dir = dataset_opt.data_dir
        self.validate_fns = glob.glob(os.path.join(self.data_dir, '*'))
        self.validate_fns.sort()
        print('[NOTE]: fetch {} samples for training'.format(len(self.validate_fns)))

    def __getitem__(self, index):
        fn = self.validate_fns[index]
        base_fn = os.path.basename(fn)
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        im = torch.tensor(im, dtype=torch.float32)
        if len(im.shape) == 2:
            im = im.unsqueeze(-1).repeat(1, 1, 3)
        im = im.permute(2, 0, 1)
        return (base_fn, im)

    def __len__(self):
        return len(self.validate_fns)
