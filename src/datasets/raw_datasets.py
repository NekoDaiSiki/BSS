import os
import glob
import torch
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset

class DataLoader_SIDD_Medium_Raw(Dataset):

    def __init__(self, dataset_opt):
        super(DataLoader_SIDD_Medium_Raw, self).__init__()
        self.data_dir = dataset_opt.data_dir
        self.patch = dataset_opt.patch
        self.train_fns = glob.glob(os.path.join(self.data_dir, '*'))
        self.train_fns.sort()
        print('fetch {} samples for training'.format(len(self.train_fns)))

    def __getitem__(self, index):
        fn = self.train_fns[index]
        im = loadmat(fn)['x']
        H, W = im.shape
        rnd_h = np.random.randint(0, max(0, H - self.patch))
        rnd_w = np.random.randint(0, max(0, W - self.patch))
        im = im[rnd_h:rnd_h + self.patch, rnd_w:rnd_w + self.patch]
        im = im[np.newaxis, :, :]
        im = torch.from_numpy(im)
        return im

    def __len__(self):
        return len(self.train_fns)

class DataLoader_SIDD_Medium_Raw_mod2(Dataset):

    def __init__(self, dataset_opt):
        super(DataLoader_SIDD_Medium_Raw_mod2, self).__init__()
        self.data_dir = dataset_opt.data_dir
        self.patch = dataset_opt.patch
        self.train_fns = glob.glob(os.path.join(self.data_dir, '*'))
        self.train_fns.sort()
        print('fetch {} samples for training'.format(len(self.train_fns)))

    def __getitem__(self, index):
        fn = self.train_fns[index]
        im = loadmat(fn)['x']
        H, W = im.shape
        rnd_h = np.random.randint(0, max(0, H - self.patch))
        rnd_h = rnd_h // 2 * 2
        rnd_w = np.random.randint(0, max(0, W - self.patch))
        rnd_w = rnd_w // 2 * 2
        im = im[rnd_h:rnd_h + self.patch, rnd_w:rnd_w + self.patch]
        im = im[np.newaxis, :, :]
        im = torch.from_numpy(im)
        return im

    def __len__(self):
        return len(self.train_fns)

class Val_SIDD_Medium_Raw(Dataset):

    def __init__(self, dataset_opt):
        super(Val_SIDD_Medium_Raw, self).__init__()
        self.data_dir = dataset_opt.data_dir
        val_data_dict = loadmat(os.path.join(self.data_dir, 'ValidationNoisyBlocksRaw.mat'))
        self.val_data_noisy = val_data_dict['ValidationNoisyBlocksRaw']
        val_data_dict = loadmat(os.path.join(self.data_dir, 'ValidationGtBlocksRaw.mat'))
        self.val_data_gt = val_data_dict['ValidationGtBlocksRaw']
        self.num_img, self.num_block, _, _ = self.val_data_gt.shape
        self.len_all = self.num_img * self.num_block
        print('fetch {} samples for testing'.format(self.num_img))

    def __getitem__(self, index):
        n_idx = index // self.num_block
        b_idx = index % self.num_block
        gt = self.val_data_gt[n_idx, b_idx][np.newaxis, :, :]
        gt = torch.from_numpy(gt)
        noisy = self.val_data_noisy[n_idx, b_idx][np.newaxis, :, :]
        noisy = torch.from_numpy(noisy)
        base_name = f'{n_idx}-{b_idx}'
        return (base_name, {'gt': gt, 'noisy': noisy})

    def __len__(self):
        return self.len_all
