import os
import time

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.main_model.base_model import BaseModel
from src.utils.register import build_from_cfg
from src.utils.builder import BACKBONE
from src.metrics.basic import calculate_psnr_from_tensor, calculate_psnr, calculate_ssim
from src.utils.img_convert import tensor2image
from src.utils.utils import mkdir


class FreqPrior_ATBSN_RealNoise(BaseModel):
    """Base class for ATBSN on real noise with frequency-prior loss.

    Replaces B2UB masking with ATBSN architecture-level blind spot.
    Data is SIDD sRGB in [0, 255] range — no synthetic noise injection.
    """
    def __init__(self, opt):
        super().__init__(opt)
        self.train_hole_size = self.main_model_opt.train_hole_size
        self.val_hole_size = self.main_model_opt.val_hole_size
        self.exp_hole_size = self.main_model_opt.exp_hole_size \
            if self.main_model_opt.exp_hole_size is not None else 3

    def create_models(self):
        self.backbone = build_from_cfg(self.sub_models_opt.backbone, BACKBONE)
        self.anchor = build_from_cfg(self.sub_models_opt.anchor, BACKBONE)
        self.sub_models.update({
            'backbone': self.backbone,
            'anchor': self.anchor,
        })

    # ---- training ----

    def train_init(self):
        self.time_start = time.time()
        noisy = self.data['noisy']
        gt = self.data['gt']
        self.data = gt
        self.noisy = noisy
        self.target = noisy

    def train_main_process(self):
        self.denoise = self.backbone(self.noisy, hole_size=self.train_hole_size)
        with torch.no_grad():
            self.exp_denoise = self.backbone(self.noisy, hole_size=self.exp_hole_size)
            self.anchor_exp = self.anchor(self.noisy, hole_size=self.val_hole_size)

    def get_beta(self):
        Lambda = (self.epoch + 1)
        Thread1 = self.loss_params.thread1
        Thread2 = self.loss_params.thread2
        Lambda1 = self.loss_params.alpha
        Lambda2 = self.loss_params.beta
        increase_ratio = self.loss_params.increase_ratio

        if Lambda <= Thread1:
            beta = Lambda2
        elif Thread1 <= Lambda <= Thread2:
            beta = Lambda2 + (Lambda - Thread1) * \
                (increase_ratio - Lambda2) / (Thread2 - Thread1)
        else:
            beta = increase_ratio

        alpha = Lambda1
        return alpha, beta

    def std(self, img, window_size=7):
        assert window_size % 2 == 1
        pad = window_size // 2
        N, C, H, W = img.shape
        img = F.pad(img, [pad] * 4, mode='reflect')
        img = F.unfold(img, kernel_size=window_size)
        img = img.view(N, C, window_size * window_size, H, W)
        img = img - torch.mean(img, dim=2, keepdim=True)
        img = img * img
        img = torch.mean(img, dim=2, keepdim=True)
        img = torch.sqrt(img)
        img = img.squeeze(2)
        return img

    def generate_soft_multiplier(self, input, lower=1, mullower=1,
                                 upper=30, mulupper=0, ws=7):
        N, C, H, W = input.shape
        ratio = input.new_ones((N, 1, H, W))
        input_std = self.std(input, ws)
        ratio[input_std < lower] = (mullower - 1) * 2 * (
            0.5 - torch.sigmoid((input_std - lower))
        )[input_std < lower] + 1
        ratio[input_std > upper] = (1 - mulupper) * 2 * (
            0.5 - torch.sigmoid((input_std - upper))
        )[input_std > upper] + 1
        ratio = ratio.detach()
        return ratio

    def generate_upper(self, input, upper=5, ws=7):
        N, C, H, W = input.shape
        ratio = input.new_ones((N, 1, H, W)) * 0
        input_std = self.std(input, ws)
        ratio[input_std > upper] = 1
        ratio = ratio.detach()
        return ratio

    def train_cal_loss(self):
        scale = 255.0 * 255.0
        diff = self.denoise - self.target
        anchor_diff = (self.anchor_exp - self.target) * self.denoise
        alpha, beta = self.get_beta()
        self.beta = beta
        self.loss_reg = alpha * torch.mean(diff ** 2) / scale
        self.loss_rev = beta * torch.mean(anchor_diff) / scale

        loss_mode = self.train_opt.loss_params.mode
        if loss_mode == 'base':
            pass
        elif loss_mode == 'freq_reg_soft_multiplier':
            lower = self.loss_params.lower
            mullower = self.loss_params.mullower
            upper = self.loss_params.upper
            mulupper = self.loss_params.mulupper
            ws = self.loss_params.ws if self.loss_params.ws else 7
            exp_mean = torch.mean(self.exp_denoise, dim=1, keepdim=True)
            freq_mask = self.generate_soft_multiplier(
                exp_mean, lower=lower, mullower=mullower,
                upper=upper, mulupper=mulupper, ws=ws)
            self.loss_reg = alpha * torch.mean(freq_mask * (diff ** 2)) / scale

        self.loss = self.loss_reg + self.loss_rev

    def train_logging(self):
        if self.curr_iter % self.train_opt.print_freq == 0:
            self.get_msg()
            self.logger.info(self.msg)

    def get_msg(self):
        lr = self.scheduler.get_last_lr()[0]
        time_end = time.time()
        ori = self.data / 255.0
        dn = self.denoise / 255.0
        dn_exp = self.exp_denoise / 255.0
        anchor = self.anchor_exp / 255.0
        distortion = calculate_psnr_from_tensor(ori, dn)
        distortion_exp = calculate_psnr_from_tensor(ori, dn_exp)
        anchor_psnr = calculate_psnr_from_tensor(ori, anchor)
        self.msg = (
            '{:04d} {:05d} lr={:.2e} dn={:.4f}, exp={:.4f}, '
            'anchor={:.4f}, Reg={:.6f}, Beta={}, Rev={:.6f}, '
            'All={:.6f}, T={:.2f}s'
        ).format(
            self.epoch, self.curr_iter, lr, distortion, distortion_exp,
            anchor_psnr, self.loss_reg.item(), self.beta,
            self.loss_rev.item(), self.loss.item(),
            time_end - self.time_start)

    # ---- validation ----

    def validate_init(self):
        self.noisy = self.data['noisy']
        self.data = self.data['gt']

    def validate_main_process(self):
        with torch.no_grad():
            self.exp_denoise = self.backbone(
                self.noisy, hole_size=self.val_hole_size)
            self.anchor_exp = self.anchor(
                self.noisy, hole_size=self.val_hole_size)

    def validate_process_images(self):
        denoise_exp = self.exp_denoise / 255.0
        data = self.data / 255.0
        noisy = self.noisy / 255.0

        self.denoise01_exp = denoise_exp.permute(
            0, 2, 3, 1).cpu().data.clamp(0, 1).numpy().squeeze(0)
        self.ori01 = data.permute(
            0, 2, 3, 1).cpu().data.clamp(0, 1).numpy().squeeze(0)

        self.denoise255_exp = tensor2image(denoise_exp)
        self.ori255 = tensor2image(data)
        self.noisy255 = tensor2image(noisy)

        anchor_exp = self.anchor_exp / 255.0
        self.anchor01_exp = anchor_exp.permute(
            0, 2, 3, 1).cpu().data.clamp(0, 1).numpy().squeeze(0)
        self.anchor255_exp = tensor2image(anchor_exp)

    def validate_cal_metrics(self, repeat, index):
        curr_validate_result = self.get_val_result(repeat, index)
        self.logger.info(curr_validate_result) if self.is_test else \
            print(curr_validate_result)
        self.validate_results.append(curr_validate_result)

    def get_val_result(self, repeat, index):
        exp_psnr = calculate_psnr(
            self.ori01.astype(np.float32),
            self.denoise01_exp.astype(np.float32), 1)
        exp_ssim = calculate_ssim(
            self.ori01.astype(np.float32) * 255,
            self.denoise01_exp.astype(np.float32) * 255)
        anchor_psnr = calculate_psnr(
            self.ori01.astype(np.float32),
            self.anchor01_exp.astype(np.float32), 1)
        anchor_ssim = calculate_ssim(
            self.ori01.astype(np.float32) * 255,
            self.anchor01_exp.astype(np.float32) * 255)

        data_name = "{:03d}-{:03d}-{:03d}_{}".format(
            index, repeat, self.epoch, self.data_name[0])
        return {
            'img_name': data_name,
            'denoise_exp_psnr': exp_psnr,
            'denoise_exp_ssim': exp_ssim,
            'anchor_exp_psnr': anchor_psnr,
            'anchor_exp_ssim': anchor_ssim,
        }

    def logger_val_summary(self, avg_results):
        self.logger.info(
            "epoch:{},exp:{:.6f}/{:.6f},anchor:{:.6f}/{:.6f}".format(
                self.epoch,
                avg_results['denoise_exp_psnr'],
                avg_results['denoise_exp_ssim'],
                avg_results['anchor_exp_psnr'],
                avg_results['anchor_exp_ssim']))

    def validate_summary(self):
        len_results = len(self.validate_results)
        avg_results = {}
        ignore_keys = ['img_name']
        for key in self.validate_results[0].keys():
            if key not in ignore_keys:
                avg_results[key] = 0
        for result in self.validate_results:
            for k in avg_results:
                avg_results[k] += result[k]
        for k in avg_results:
            avg_results[k] /= len_results
        self.logger_val_summary(avg_results)
        self.validate_results = []

    def save_results(self, repeat=0, idx=0):
        if (self.is_train and self.validate_opt.save_results) or \
           (self.is_test and self.test_opt.save_results):
            out = 'val_imgs' if self.is_train else 'test_imgs'
            opt_tv = self.datasets_opt.validate if self.is_train \
                else self.datasets_opt.test
            name_dataset = opt_tv.dataset_path.split('/')[-1]
            save_dir = os.path.join(self.train_url, out, name_dataset)
            mkdir(save_dir)

            save_path = os.path.join(
                save_dir,
                "{:03d}-{:03d}-{:03d}_exp.png".format(idx, repeat, self.epoch))
            Image.fromarray(self.denoise255_exp).convert('RGB').save(save_path)

            save_path = os.path.join(
                save_dir,
                "{:03d}-{:03d}-{:03d}_anchor.png".format(idx, repeat, self.epoch))
            Image.fromarray(self.anchor255_exp).convert('RGB').save(save_path)

            save_path = os.path.join(
                save_dir,
                "{:03d}-{:03d}-{:03d}_noisy.png".format(idx, repeat, self.epoch))
            Image.fromarray(self.noisy255).convert('RGB').save(save_path)


class FreqPrior_ATBSN_RealNoise_SupStd(FreqPrior_ATBSN_RealNoise):
    """Adds SupStd mean-loss on top of freq-prior ATBSN."""

    def meanPerChannel(self, img, window_size=3):
        assert window_size % 2 == 1
        pad = window_size // 2
        N, C, H, W = img.shape
        img = F.pad(img, [pad] * 4, mode='reflect')
        img = F.unfold(img, kernel_size=window_size)
        img = img.view(N, C, window_size * window_size, H, W)
        if self.train_opt.clear_center:
            img = torch.cat([
                img[:, :, :window_size**2 // 2, ...],
                img[:, :, window_size**2 // 2 + 1:, ...]
            ], dim=2)
        img_mean = torch.mean(img, dim=2)
        return img_mean

    def calMeanLoss(self, direct_dn, exp_dn):
        scale = 255.0 * 255.0
        parms = self.train_opt.loss_params.mean_loss_params
        weight = parms.weight
        ws = parms.ws
        thres = parms.thres

        mean_exp = self.meanPerChannel(exp_dn, ws)
        mean_loss = (direct_dn - mean_exp) ** 2

        mask_input = torch.mean(exp_dn, dim=1, keepdim=True)
        mask = 1 - self.generate_upper(mask_input, thres, ws)

        return weight * torch.mean(mask * mean_loss) / scale

    def train_main_process(self):
        super().train_main_process()
        self.dirct_denoise = self.backbone(self.noisy, hole_size=self.exp_hole_size)

    def train_cal_loss(self):
        super().train_cal_loss()
        self.loss_mean = self.beta * self.calMeanLoss(
            self.dirct_denoise, self.anchor_exp)
        self.loss = self.loss_reg + self.loss_rev + self.loss_mean

    def get_msg(self):
        lr = self.scheduler.get_last_lr()[0]
        time_end = time.time()
        ori = self.data / 255.0
        dn = self.denoise / 255.0
        dn_exp = self.exp_denoise / 255.0
        anchor = self.anchor_exp / 255.0
        distortion = calculate_psnr_from_tensor(ori, dn)
        distortion_exp = calculate_psnr_from_tensor(ori, dn_exp)
        anchor_psnr = calculate_psnr_from_tensor(ori, anchor)
        self.msg = (
            '{:04d} {:05d} lr={:.2e} dn={:.4f}, exp={:.4f}, '
            'anchor={:.4f}, Reg={:.6f}, Beta={}, Rev={:.6f}, '
            'Mean={:.6f}, All={:.6f}, T={:.2f}s'
        ).format(
            self.epoch, self.curr_iter, lr, distortion, distortion_exp,
            anchor_psnr, self.loss_reg.item(), self.beta,
            self.loss_rev.item(), self.loss_mean, self.loss.item(),
            time_end - self.time_start)


class FreqPrior_ATBSN_RealNoise_SupStd_AnchorMomentum(
        FreqPrior_ATBSN_RealNoise_SupStd):
    """Full pipeline: ATBSN + freq prior + SupStd + anchor EMA momentum."""

    def __init__(self, opt):
        super().__init__(opt)
        self.m = self.main_model_opt.m

    @torch.no_grad()
    def _momentum_update(self, encoder_q, encoder_k):
        for param_q, param_k in zip(
            encoder_q.parameters(), encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def train_main_process(self):
        self._momentum_update(self.backbone, self.anchor)
        super().train_main_process()
