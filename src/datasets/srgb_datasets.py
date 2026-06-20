import os
import glob
import torch
import imageio

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImagenetValDataset(Dataset):
    def __init__(self, dataset_opt):
        super(ImagenetValDataset, self).__init__()
        self.data_dir = dataset_opt.data_dir
        self.patch = dataset_opt.patch
        self.train_fns = glob.glob(os.path.join(self.data_dir, "*"))
        self.train_fns.sort()
        print('[NOTE]: fetch {} samples for training'.format(len(self.train_fns)))

    def __getitem__(self, index):
        # fetch image
        fn = self.train_fns[index]
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        # random crop
        H = im.shape[0]
        W = im.shape[1]
        if H - self.patch > 0:
            xx = np.random.randint(0, H - self.patch)
            im = im[xx:xx + self.patch, :, :]
        if W - self.patch > 0:
            yy = np.random.randint(0, W - self.patch)
            im = im[:, yy:yy + self.patch, :]
        # np.ndarray to torch.tensor
        # return a tensor of size (C, H, W), dtype torch.float32, range [0.,255.]
        im = torch.tensor(im,dtype=torch.float32).permute(2,0,1)
        return im

    def __len__(self):
        return len(self.train_fns)
    
class BasicValidateDataset(Dataset):
    def __init__(self, dataset_opt):
        self.data_dir = dataset_opt.data_dir
        self.validate_fns = glob.glob(os.path.join(self.data_dir, "*"))
        self.validate_fns.sort()
        print('[NOTE]: fetch {} samples for training'.format(len(self.validate_fns)))
    
    def __getitem__(self, index):
        # fetch image
        fn = self.validate_fns[index]
        base_fn = os.path.basename(fn)
        im = Image.open(fn)
        im = np.array(im, dtype=np.float32)
        # np.ndarray to torch.tensor
        im = torch.tensor(im,dtype=torch.float32)

        # 对黑白图片特殊处理
        if len(im.shape) == 2:
            im = im.unsqueeze(-1).repeat(1,1,3)
        im = im.permute(2,0,1)
        return base_fn,im

    def __len__(self):
        return len(self.validate_fns)
