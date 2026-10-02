"""Shared input preparation and validation loop for the BSS models."""
import time
import torch
import torch.nn.functional as F
from .base_model import BaseModel
from src.utils.register import build_from_cfg
from src.utils.builder import BACKBONE, NOISER

class Noise2Clean(BaseModel):

    def __init__(self, opt):
        super(Noise2Clean, self).__init__(opt)

    def create_models(self):
        if 'noiser' in self.sub_models_opt:
            self.noise_adder = build_from_cfg(self.sub_models_opt.noiser, NOISER)
        self.backbone = build_from_cfg(self.sub_models_opt.backbone, BACKBONE)
        self.sub_models.update({'backbone': self.backbone})

    def train_init(self):
        self.time_start = time.time()
        self.data = self.data / 255.0
        self.noisyA = self.noise_adder.add_noise(self.data)
        self.input = self.noisyA
        self.target = self.data

    def train_logging(self):
        if self.curr_iter % self.train_opt.print_freq == 0:
            self.get_msg()
            self.logger.info(self.msg)

    def validate_init(self):
        self.data = self.data / 255.0
        self.noisy = self.noise_adder.add_noise(self.data)
        n, c, h, w = self.data.shape
        self.H, self.W = (h, w)
        val_size_h = (h + 31) // 32 * 32
        val_size_w = (w + 31) // 32 * 32
        self.noisy = F.pad(self.noisy, [0, val_size_w - w, 0, val_size_h - h], mode='reflect')
        self.input = self.noisy

    def validate_cal_metrics(self, repeat, index):
        curr_validate_result = self.get_val_result(repeat, index)
        self.logger.info(curr_validate_result) if self.is_test else print('Not log details in validate')
        self.validate_results.append(curr_validate_result)

    def validate_summary(self):
        len_results = len(self.validate_results)
        avg_results = {}
        ignore_keys = ['img_name']
        for key in self.validate_results[0].keys():
            if not key in ignore_keys:
                avg_results.update({key: 0})
        for result in self.validate_results:
            for k in avg_results.keys():
                avg_results[k] += result[k]
        for k in avg_results.keys():
            avg_results[k] = avg_results[k] / len_results
        self.logger_val_summary(avg_results)
        self.validate_results = []

class Noise2Clean_SIDDsrgb_SA(Noise2Clean):

    def train_init(self):
        self.time_start = time.time()
        noisy = self.data['noisy']
        gt = self.data['gt']
        self.data = gt
        self.noisyA = noisy
        self.input = self.noisyA
        self.target = gt

    def validate_init(self):
        self.noisy = self.data['noisy']
        self.data = self.data['gt']
        n, c, h, w = self.data.shape
        self.input = self.noisy

class Noise2Clean_SIDDRaw(Noise2Clean):

    def space_to_depth(self, x, block_size):
        n, c, h, w = x.size()
        unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
        return unfolded_x.view(n, c * block_size ** 2, h // block_size, w // block_size)

    def depth_to_space(self, x, block_size):
        return torch.nn.functional.pixel_shuffle(x, block_size)

    def train_init(self):
        self.time_start = time.time()
        self.data = self.space_to_depth(self.data, 2)
        self.noisyA = self.data
        self.input = self.noisyA
        self.target = self.data

    def validate_init(self):
        self.noisy = self.data['noisy']
        self.data = self.data['gt']
        n, c, h, w = self.data.shape
        self.input = self.space_to_depth(self.noisy, block_size=2)
