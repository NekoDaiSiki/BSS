import numpy as np
import random
import torch.utils.data as data

def aug_np3(img, flip_h, flip_w, transpose):
    if flip_h:
        img = img[:, ::-1, :]
    if flip_w:
        img = img[:, :, ::-1]
    if transpose:
        img = np.transpose(img, (0, 2, 1))
    return img

def crop_np3(img, patch_size, position_h, position_w):
    return img[:, position_h:position_h + patch_size, position_w:position_w + patch_size]

class BaseTrainDataset(data.Dataset):

    def __init__(self, path, patch_size, pin_memory):
        super(BaseTrainDataset, self).__init__()
        self.patch_size = patch_size
        self.pin_memory = pin_memory
        self._get_img_paths(path)
        if self.pin_memory:
            self._open_images()

    def __len__(self):
        return 100000

    def crop(self, img_L, img_H=None):
        C, H, W = img_L.shape
        position_H = random.randint(0, H - self.patch_size)
        position_W = random.randint(0, W - self.patch_size)
        patch_L = crop_np3(img_L, self.patch_size, position_H, position_W)
        if img_H is not None:
            patch_H = crop_np3(img_H, self.patch_size, position_H, position_W)
            return (patch_L, patch_H)
        else:
            return patch_L

    def augment(self, img_L, img_H=None):
        flip_h = random.random() > 0.5
        flip_w = random.random() > 0.5
        transpose = random.random() > 0.5
        img_L = aug_np3(img_L, flip_h, flip_w, transpose)
        if img_H is not None:
            img_H = aug_np3(img_H, flip_h, flip_w, transpose)
            return (img_L, img_H)
        else:
            return img_L
