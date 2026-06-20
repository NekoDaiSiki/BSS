import os
import glob
import torch
import numpy as np
from PIL import Image
from scipy.io import loadmat, savemat

from torch.utils.data import Dataset


class DataLoader_SIDD_Medium_Raw(Dataset):
    def __init__(self, dataset_opt):
        super(DataLoader_SIDD_Medium_Raw, self).__init__()
        self.data_dir = dataset_opt.data_dir
        # get images path
        self.patch = dataset_opt.patch
        self.train_fns = glob.glob(os.path.join(self.data_dir, "*"))
        self.train_fns.sort()
        print('fetch {} samples for training'.format(len(self.train_fns)))

    def __getitem__(self, index):
        # fetch image
        fn = self.train_fns[index]
        im = loadmat(fn)["x"]
        # random crop
        H, W = im.shape
        rnd_h = np.random.randint(0, max(0, H - self.patch))
        rnd_w = np.random.randint(0, max(0, W - self.patch))
        im = im[rnd_h : rnd_h + self.patch, rnd_w : rnd_w + self.patch]
        im = im[np.newaxis, :, :]
        im = torch.from_numpy(im)
        return im

    def __len__(self):
        return len(self.train_fns)

class DataLoader_SIDD_Medium_Raw_n2c(Dataset):
    def __init__(self, dataset_opt):
        super(DataLoader_SIDD_Medium_Raw_n2c, self).__init__()
        self.data_dir = dataset_opt.data_dir
        self.ny_dir = os.path.join(self.data_dir,'ny')
        self.gt_dir = os.path.join(self.data_dir,'gt')

        # get images path
        self.patch = dataset_opt.patch
        self.train_fns = glob.glob(os.path.join(self.ny_dir, "*"))
        self.train_fns.sort()

        print('fetch {} samples for training'.format(len(self.train_fns)))

    def __getitem__(self, index):
        # fetch image
        fn_ny = self.train_fns[index]
        im_ny = loadmat(fn_ny)["x"]

        fn_gt = os.path.join(self.gt_dir,fn_ny.split('/')[-1])
        im_gt = loadmat(fn_gt)["x"]
        # random crop
        H, W = im_ny.shape
        rnd_h = np.random.randint(0, max(0, H - self.patch))
        rnd_w = np.random.randint(0, max(0, W - self.patch))
        im_ny = im_ny[rnd_h : rnd_h + self.patch, rnd_w : rnd_w + self.patch]
        im_ny = im_ny[np.newaxis, :, :]
        im_ny = torch.from_numpy(im_ny)

        im_gt = im_gt[rnd_h : rnd_h + self.patch, rnd_w : rnd_w + self.patch]
        im_gt = im_gt[np.newaxis, :, :]
        im_gt = torch.from_numpy(im_gt)

        base_name = self.gt_dir,fn_ny.split('/')[-1]
        return base_name, {'noisy':im_ny,'gt': im_gt}

    def __len__(self):
        return len(self.train_fns)


class DataLoader_SIDD_Medium_Raw_n2c_mod2(DataLoader_SIDD_Medium_Raw_n2c):
    def __getitem__(self, index):
        # fetch image
        fn_ny = self.train_fns[index]
        im_ny = loadmat(fn_ny)["x"]

        fn_gt = os.path.join(self.gt_dir,fn_ny.split('/')[-1])
        im_gt = loadmat(fn_gt)["x"]
        # random crop
        H, W = im_ny.shape
        rnd_h = np.random.randint(0, max(0, H - self.patch))
        rnd_h = rnd_h//2*2
        rnd_w = np.random.randint(0, max(0, W - self.patch))
        rnd_w = rnd_w//2*2
        im_ny = im_ny[rnd_h : rnd_h + self.patch, rnd_w : rnd_w + self.patch]
        im_ny = im_ny[np.newaxis, :, :]
        im_ny = torch.from_numpy(im_ny)

        im_gt = im_gt[rnd_h : rnd_h + self.patch, rnd_w : rnd_w + self.patch]
        im_gt = im_gt[np.newaxis, :, :]
        im_gt = torch.from_numpy(im_gt)

        base_name = self.gt_dir,fn_ny.split('/')[-1]
        return base_name, {'noisy':im_ny,'gt': im_gt}


class DataLoader_SIDD_Medium_Raw_mod2(Dataset):
    def __init__(self, dataset_opt):
        super(DataLoader_SIDD_Medium_Raw_mod2, self).__init__()
        self.data_dir = dataset_opt.data_dir
        # get images path
        self.patch = dataset_opt.patch
        self.train_fns = glob.glob(os.path.join(self.data_dir, "*"))
        self.train_fns.sort()
        print('fetch {} samples for training'.format(len(self.train_fns)))

    def __getitem__(self, index):
        # fetch image
        fn = self.train_fns[index]
        im = loadmat(fn)["x"]
        # random crop
        H, W = im.shape
        rnd_h = np.random.randint(0, max(0, H - self.patch))
        rnd_h = rnd_h//2*2
        rnd_w = np.random.randint(0, max(0, W - self.patch))
        rnd_w = rnd_h//2*2
        im = im[rnd_h : rnd_h + self.patch, rnd_w : rnd_w + self.patch]
        im = im[np.newaxis, :, :]
        im = torch.from_numpy(im)
        return im

    def __len__(self):
        return len(self.train_fns)

class Val_SIDD_Medium_Raw(Dataset):
    def __init__(self, dataset_opt):
        super(Val_SIDD_Medium_Raw, self).__init__()
        self.data_dir = dataset_opt.data_dir
        # get images path
        # self.train_fns = glob.glob(os.path.join(self.data_dir, "*"))
        # self.train_fns.sort()
        val_data_dict = loadmat(
            os.path.join(self.data_dir, "ValidationNoisyBlocksRaw.mat"))
        self.val_data_noisy = val_data_dict['ValidationNoisyBlocksRaw']

        val_data_dict = loadmat(
            os.path.join(self.data_dir, 'ValidationGtBlocksRaw.mat'))
        self.val_data_gt = val_data_dict['ValidationGtBlocksRaw']
        self.num_img, self.num_block, _, _ = self.val_data_gt.shape
        self.len_all = self.num_img*self.num_block

        print('fetch {} samples for testing'.format(self.num_img))


    def __getitem__(self, index):
        # fetch image
        n_idx = index//self.num_block
        b_idx = index%self.num_block

        gt = self.val_data_gt[n_idx,b_idx][np.newaxis,:,:]
        gt = torch.from_numpy(gt)
        noisy = self.val_data_noisy[n_idx,b_idx][np.newaxis,:,:]
        noisy = torch.from_numpy(noisy)

        base_name = f'{n_idx}-{b_idx}' 
        return base_name, {'gt':gt,'noisy':noisy}

    def __len__(self):
        return self.len_all



class Val_SIDD_Benchmark(Dataset):
    def __init__(self, dataset_opt):
        super(Val_SIDD_Benchmark, self).__init__()
        self.data_dir = dataset_opt.data_dir
        # get images path
        # self.train_fns = glob.glob(os.path.join(self.data_dir, "*"))
        # self.train_fns.sort()
        val_data_dict = loadmat(
            os.path.join(self.data_dir, "BenchmarkNoisyBlocksRaw.mat"))
        self.val_data_noisy = val_data_dict['BenchmarkNoisyBlocksRaw']

        self.num_img, self.num_block, _, _ = self.val_data_gt.shape
        self.len_all = self.num_img*self.num_block

        print('fetch {} samples for testing'.format(self.num_img))


    def __getitem__(self, index):
        # fetch image
        n_idx = index//self.num_block
        b_idx = index%self.num_block
        
        noisy = self.val_data_noisy[n_idx,b_idx][np.newaxis,:,:]
        noisy = torch.from_numpy(noisy)

        base_name = f'{n_idx}-{b_idx}' 
        return base_name, {'noisy':noisy}

    def __len__(self):
        return self.len_all