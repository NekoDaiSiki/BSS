import os
import time
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.register import build_from_cfg
from src.utils.builder import BACKBONE
from src.metrics.basic import calculate_psnr_from_tensor, calculate_psnr, calculate_ssim
from src.utils.img_convert import tensor2image
from src.utils.utils import mkdir
from .blind_spot import B2UB_Model

class Frequency_B2UB_Model(B2UB_Model):

    def create_models(self):
        super().create_models()
        self.anchor = build_from_cfg(self.sub_models_opt.anchor, BACKBONE)
        self.sub_models.update({'anchor': self.anchor})

    def get_beta(self):
        Lambda = self.epoch + 1
        Thread1 = self.loss_params.thread1
        Thread2 = self.loss_params.thread2
        Lambda1 = self.loss_params.alpha
        Lambda2 = self.loss_params.beta
        increase_ratio = self.loss_params.increase_ratio
        if Lambda <= Thread1:
            beta = Lambda2
        elif Thread1 <= Lambda <= Thread2:
            beta = Lambda2 + (Lambda - Thread1) * (increase_ratio - Lambda2) / (Thread2 - Thread1)
        else:
            beta = increase_ratio
        alpha = Lambda1
        return (alpha, beta)

    def train_main_process(self):
        super().train_main_process()
        with torch.no_grad():
            self.anchor_exp = self.anchor(self.noisy)

    def std(self, img, window_size=7):
        assert window_size % 2 == 1
        pad = window_size // 2
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

    def generate_soft_multiplier(self, input, lower=1, mullower=1, upper=30, mulupper=0, ws=7):
        N, C, H, W = input.shape
        ratio = input.new_ones((N, 1, H, W)) * 1
        input_std = self.std(input, ws)
        low = input_std < lower
        high = input_std > upper
        ratio[low] = 1 - (mullower - 1) * torch.tanh(input_std[low] - lower)
        ratio[high] = 1 - (1 - mulupper) * torch.tanh(input_std[high] - upper)
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
        diff = self.denoise - self.target
        exp_diff = (self.exp_denoise - self.target) * self.denoise
        alpha, beta = self.get_beta()
        revisible = exp_diff
        self.beta = beta
        self.loss_reg = alpha * torch.mean(diff ** 2)
        self.loss_rev = beta * torch.mean(revisible)
        loss_mode = self.train_opt.loss_params.mode
        if loss_mode == 'freq_reg_soft_multiplier':
            lower = self.loss_params.lower
            mullower = self.loss_params.mullower
            upper = self.loss_params.upper
            mulupper = self.loss_params.mulupper
            ws = self.loss_params.ws if self.loss_params.ws else 7
            exp_mean = torch.mean(self.exp_denoise, dim=1, keepdim=True) * 255.0
            freq_mask = self.generate_soft_multiplier(exp_mean, lower=lower, mullower=mullower, upper=upper, mulupper=mulupper, ws=ws)
            self.loss_reg = alpha * torch.mean(freq_mask * diff ** 2)
        else:
            raise ValueError('Unsupported NSC-BSM loss mode: {}'.format(loss_mode))
        self.loss = self.loss_reg + self.loss_rev

    def get_msg(self):
        lr = self.scheduler.get_last_lr()[0]
        time_start = self.time_start
        time_end = time.time()
        dn = self.denoise
        dn_exp = self.exp_denoise
        ori = self.data
        distortion = calculate_psnr_from_tensor(ori, dn)
        distortion_exp = calculate_psnr_from_tensor(ori, dn_exp)
        anchor_exp = calculate_psnr_from_tensor(ori, self.anchor_exp)
        self.msg = '{:04d} {:05d} lr={:.2e} distortion={:.6f}, distortion_exp={:.6f}, anchor_exp={:.6f},Loss_Reg={:.6f}, Beta={}, Loss_Rev={:.6f}, Loss_All={:.6f}, Time={:.4f}'.format(self.epoch, self.curr_iter, lr, distortion, distortion_exp, anchor_exp, self.loss_reg.item(), self.beta, self.loss_rev.item(), self.loss.item(), time_end - time_start)

    def validate_init(self):
        self.data = self.data / 255.0
        self.noisy = self.train_noise_adder.add_noise(self.data)
        n, c, h, w = self.data.shape
        self.H, self.W = (h, w)
        val_size = (max(h, w) + 31) // 32 * 32
        self.noisy = F.pad(self.noisy, [0, val_size - w, 0, val_size - h], mode='reflect')

    def validate_main_process(self):
        with torch.no_grad():
            self.exp_denoise = self.backbone(self.noisy)

    def validate_process_images(self):
        denoise_exp = self.exp_denoise[:, :, :self.H, :self.W]
        self.denoise255_exp = tensor2image(denoise_exp)
        self.ori255 = tensor2image(self.data)
        self.noisy255 = tensor2image(self.noisy[:, :, :self.H, :self.W])

    def get_val_result(self, repeat, index):
        denoise_exp_psnr = calculate_psnr(self.ori255, self.denoise255_exp)
        denoise_exp_ssim = calculate_ssim(self.ori255, self.denoise255_exp)
        data_name = '{:03d}-{:03d}-{:03d}_{}'.format(index, repeat, self.epoch, self.data_name[0])
        curr_validate_result = {'img_name': data_name, 'denoise_exp_psnr': denoise_exp_psnr, 'denoise_exp_ssim': denoise_exp_ssim}
        return curr_validate_result

    def save_results(self, repeat=0, idx=0):
        if self.is_train and self.validate_opt.save_results or (self.is_test and self.test_opt.save_results):
            out = 'val_imgs' if self.is_train else 'test_imgs'
            opt_testOrVal = self.datasets_opt.validate if self.is_train else self.datasets_opt.test
            name_dataset = opt_testOrVal.data_dir.split('/')[-1]
            save_dir = os.path.join(self.train_url, out, name_dataset)
            mkdir(save_dir)
            save_path = os.path.join(save_dir, '{:03d}-{:03d}-{:03d}_exp.png'.format(idx, repeat, self.epoch))
            Image.fromarray(self.denoise255_exp).convert('RGB').save(save_path)
            save_path = os.path.join(save_dir, '{:03d}-{:03d}-{:03d}_noisy.png'.format(idx, repeat, self.epoch))
            Image.fromarray(self.noisy255).convert('RGB').save(save_path)

    def logger_val_summary(self, avg_results):
        self.logger.info('epoch:{},exp:{:.6f}/{:.6f}'.format(self.epoch, avg_results['denoise_exp_psnr'], avg_results['denoise_exp_ssim']))

class Frequency_B2UB_Model_SupStd(Frequency_B2UB_Model):

    def meanPerChannel(self, img, window_size=3):
        assert window_size % 2 == 1
        pad = window_size // 2
        N, C, H, W = img.shape
        img = nn.functional.pad(img, [pad] * 4, mode='reflect')
        img = nn.functional.unfold(img, kernel_size=window_size)
        img = img.view(N, C, window_size * window_size, H, W)
        if self.train_opt.clear_center:
            img = torch.cat([img[:, :, :window_size ** 2 // 2, ...], img[:, :, window_size ** 2 // 2 + 1:, ...]], dim=2)
        img_mean = torch.mean(img, dim=2)
        return img_mean

    def calMeanLoss(self, direct_dn, exp_dn):
        parms = self.train_opt.loss_params.mean_loss_params
        mean_ws = parms.mean_ws if 'mean_ws' in parms else parms.ws
        mean_exp = self.meanPerChannel(exp_dn, mean_ws)
        mean_loss = (direct_dn - mean_exp) ** 2
        mask_input = torch.mean(direct_dn.detach(), dim=1, keepdim=True) * 255.0
        mask = 1 - self.generate_upper(mask_input, parms.thres, parms.ws)
        return parms.weight * torch.mean(mask * mean_loss)

    def train_main_process(self):
        super().train_main_process()
        self.dirct_denoise = self.backbone(self.noisy)

    def get_msg(self):
        lr = self.scheduler.get_last_lr()[0]
        time_start = self.time_start
        time_end = time.time()
        dn = self.denoise
        dn_exp = self.exp_denoise
        ori = self.data
        distortion = calculate_psnr_from_tensor(ori, dn)
        distortion_exp = calculate_psnr_from_tensor(ori, dn_exp)
        anchor_exp = calculate_psnr_from_tensor(ori, self.anchor_exp)
        self.msg = '{:04d} {:05d} lr={:.2e} distortion={:.6f}, distortion_exp={:.6f}, anchor_exp={:.6f},Loss_Reg={:.6f}, Beta={}, Loss_Rev={:.6f}, Loss_Mean={:.6f}, Loss_All={:.6f}, Time={:.4f}'.format(self.epoch, self.curr_iter, lr, distortion, distortion_exp, anchor_exp, self.loss_reg.item(), self.beta, self.loss_rev.item(), self.loss_mean, self.loss.item(), time_end - time_start)

class Frequency_B2UB_Model_SupStd_Anchor(Frequency_B2UB_Model_SupStd):

    def train_cal_loss(self):
        super().train_cal_loss()
        self.loss_mean = self.calMeanLoss(self.dirct_denoise, self.anchor_exp)
        self.loss = self.loss_reg + self.loss_rev + self.loss_mean

class Frequency_B2UB_Model_SupStd_AnchorMomentum(Frequency_B2UB_Model_SupStd_Anchor):

    def __init__(self, opt):
        super().__init__(opt)
        self.m = self.main_model_opt.m

    @torch.no_grad()
    def _momentum_update(self, encoder_q, encoder_k):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def train_main_process(self):
        self._momentum_update(self.backbone, self.anchor)
        super().train_main_process()

    def validate_main_process(self):
        super().validate_main_process()
        with torch.no_grad():
            self.anchor_exp = self.anchor(self.noisy)

    def validate_process_images(self):
        super().validate_process_images()
        anchor_exp = self.anchor_exp[:, :, :self.H, :self.W]
        self.anchor255_exp = tensor2image(anchor_exp)

    def get_val_result(self, repeat, index):
        curr_validate_result = super().get_val_result(repeat, index)
        anchor_exp_psnr = calculate_psnr(self.ori255, self.anchor255_exp)
        anchor_exp_ssim = calculate_ssim(self.ori255, self.anchor255_exp)
        curr_validate_result.update({'anchor_exp_psnr': anchor_exp_psnr, 'anchor_exp_ssim': anchor_exp_ssim})
        print(curr_validate_result)
        return curr_validate_result

    def logger_val_summary(self, avg_results):
        self.logger.info('epoch:{},exp:{:.6f}/{:.6f},anchor:{:.6f}/{:.6f}'.format(self.epoch, avg_results['denoise_exp_psnr'], avg_results['denoise_exp_ssim'], avg_results['anchor_exp_psnr'], avg_results['anchor_exp_ssim']))

    def save_results(self, repeat=0, idx=0):
        super().save_results(repeat, idx)
        if self.is_train and self.validate_opt.save_results or (self.is_test and self.test_opt.save_results):
            out = 'val_imgs' if self.is_train else 'test_imgs'
            opt_testOrVal = self.datasets_opt.validate if self.is_train else self.datasets_opt.test
            name_dataset = opt_testOrVal.data_dir.split('/')[-1]
            save_dir = os.path.join(self.train_url, out, name_dataset)
            mkdir(save_dir)
            save_path = os.path.join(save_dir, '{:03d}-{:03d}-{:03d}_anchor.png'.format(idx, repeat, self.epoch))
            Image.fromarray(self.anchor255_exp).convert('RGB').save(save_path)
