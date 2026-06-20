import os
import time
import random

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.main_model.base_model import BaseModel
from src.utils.register import build_from_cfg
from src.utils.builder import MASKER,BACKBONE,NOISER,HEAD
from src.metrics.basic import calculate_psnr_from_tensor,calculate_psnr,calculate_ssim
from src.utils.img_convert import tensor2image

from .blind_spot import B2UB_Model

class B2UB_Contrast(B2UB_Model):
    def create_models(self):
        super().create_models()
        self.anchor = build_from_cfg(self.sub_models_opt.anchor,BACKBONE)
        self.sub_models.update({
            'anchor': self.anchor,
        })

    def get_beta(self):
        Lambda = (self.epoch+1)
        Thread1 = self.loss_params.thread1
        Thread2 = self.loss_params.thread2
        Lambda1 = self.loss_params.alpha
        Lambda2 = self.loss_params.beta
        increase_ratio = self.loss_params.increase_ratio

        if Lambda <= Thread1:
            beta = Lambda2
        elif Thread1 <= Lambda <= Thread2:
            beta = Lambda2 + (Lambda - Thread1) * \
                (increase_ratio-Lambda2) / (Thread2-Thread1)
        else:
            beta = increase_ratio
        
        alpha = Lambda1     

        return alpha, beta
    def train_main_process(self):
        super().train_main_process()
        with torch.no_grad():
            self.anchor_exp = self.anchor(self.noisy)

    def std(self,img, window_size=7):
        assert window_size % 2 == 1
        pad = window_size // 2
        # calculate std on the mean image of the color channels
        N, C, H, W = img.shape
        img = nn.functional.pad(img, [pad] * 4, mode='reflect')
        img = nn.functional.unfold(img, kernel_size=window_size)
        img = img.view(N, C, window_size * window_size, H, W)
        img = img - torch.mean(img, dim=2, keepdim=True)
        img = img * img
        img = torch.mean(img, dim=2, keepdim=True)
        img = torch.sqrt(img)
        img = img.squeeze(2)
        return img

    def generate_std_mask(self,input,thres,ws):
        N, C, H, W = input.shape
        ratio = input.new_ones((N, 1, H, W)) * 0
        input_std = self.std(input,ws)
        ratio[input_std < thres] = 1
        ratio = ratio.detach()
        return ratio  

    def get_contrastive_loss(self,denoise_exp):
        low_freq_loss = self.low_freq_loss(denoise_exp)
        high_freq_loss = self.high_freq_loss(denoise_exp)
        return low_freq_loss + high_freq_loss


    def low_freq_loss(self,denoise_exp):
        lf_parms = self.loss_params.lf_parms
        ws = lf_parms.ws
        thres = lf_parms.thres
        lf_mask = self.generate_std_mask(denoise_exp,thres,ws)


        return 0
    
    def high_freq_loss(self,denoise_exp):
        return 0

    def train_cal_loss(self):
        diff = self.denoise - self.target
        exp_diff = (self.exp_denoise - self.target)* self.denoise
        alpha,beta = self.get_beta()
        revisible = exp_diff
        self.beta = beta
        self.loss_reg = alpha * torch.mean(diff**2)
        self.loss_rev = beta*torch.mean(revisible)

        # sim contrastive loss
        self.contrast_loss = self.get_contrastive_loss(self.exp_denoise)
        self.loss = self.loss_reg + self.loss_rev