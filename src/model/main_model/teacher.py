"""BSS pair construction, masked supervision and evaluation for the configured noise settings."""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.io import savemat
from src.utils.builder import BACKBONE
from src.utils.register import build_from_cfg
from src.utils.img_convert import tensor2image
from src.metrics.basic import calculate_psnr_from_tensor, calculate_psnr, calculate_ssim
from src.utils.utils import mkdir
from .denoising_base import Noise2Clean, Noise2Clean_SIDDRaw

class OneTeacher(Noise2Clean):

    def create_models(self):
        super().create_models()
        self.net_teacher1 = build_from_cfg(self.sub_models_opt.teacher1, BACKBONE)
        self.sub_models.update({'teacher1': self.net_teacher1})

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

    def generate_std_mask(self, input, thres, ws=7):
        N, C, H, W = input.shape
        ratio = input.new_ones((N, 1, H, W)) * 0
        input_std = self.std(input, ws)
        ratio[input_std < thres] = 1
        ratio = ratio.detach()
        return ratio

    def validate_main_process(self):
        with torch.no_grad():
            self.teacher1 = self.net_teacher1(self.input)
            self.student = self.backbone(self.input)

    def get_msg(self):
        lr = self.scheduler.get_last_lr()[0]
        time_start = self.time_start
        time_end = time.time()
        ori = self.data
        teacher1 = calculate_psnr_from_tensor(ori, self.teacher1)
        student = calculate_psnr_from_tensor(ori, self.student)
        student_exp = calculate_psnr_from_tensor(ori, self.student_exp)
        self.msg = '{:04d} {:05d} lr={:.2e} teacher1={:.6f}, student={:.6f}, student_exp={:.6f}, Loss_All={:.6f}, Time={:.4f}'.format(self.epoch, self.curr_iter, lr, teacher1, student, student_exp, self.loss.item(), time_end - time_start)

    def validate_process_images(self):
        teacher1 = self.teacher1[:, :, :self.H, :self.W]
        student = self.student[:, :, :self.H, :self.W]
        self.teacher1_255 = tensor2image(teacher1)
        self.student_255 = tensor2image(student)
        self.ori255 = tensor2image(self.data)
        self.noisy255 = tensor2image(self.noisy[:, :, :self.H, :self.W])
        self.denoise255 = self.student_255

    def get_val_result(self, repeat, index):
        teacher1_psnr = calculate_psnr(self.ori255, self.teacher1_255)
        teacher1_ssim = calculate_ssim(self.ori255, self.teacher1_255)
        student_psnr = calculate_psnr(self.ori255, self.student_255)
        student_ssim = calculate_ssim(self.ori255, self.student_255)
        data_name = '{:03d}-{:03d}-{:03d}_{}'.format(index, repeat, self.epoch, self.data_name[0])
        curr_validate_result = {'img_name': data_name, 'teacher1_psnr': teacher1_psnr, 'teacher1_ssim': teacher1_ssim, 'student_psnr': student_psnr, 'student_ssim': student_ssim}
        return curr_validate_result

    def logger_val_summary(self, avg_results):
        self.logger.info('epoch:{},teacher1:{:.6f}/{:.6f}, student:{:.6f}/{:.6f}'.format(self.epoch, avg_results['teacher1_psnr'], avg_results['teacher1_ssim'], avg_results['student_psnr'], avg_results['student_ssim']))

    def save_results(self, repeat=0, idx=0):
        if self.is_train and self.validate_opt.save_results or (self.is_test and self.test_opt.save_results):
            out = 'val_imgs' if self.is_train else 'test_imgs'
            opt_testOrVal = self.datasets_opt.validate if self.is_train else self.datasets_opt.test
            name_dataset = opt_testOrVal.data_dir.split('/')[-1]
            save_dir = os.path.join(self.train_url, out, name_dataset)
            mkdir(save_dir)
            save_path = os.path.join(save_dir, '{:03d}-{:03d}-{:03d}_teacher.png'.format(idx, repeat, self.epoch))
            Image.fromarray(self.teacher1_255).convert('RGB').save(save_path)
            save_path = os.path.join(save_dir, '{:03d}-{:03d}-{:03d}_student.png'.format(idx, repeat, self.epoch))
            Image.fromarray(self.student_255).convert('RGB').save(save_path)
            save_path = os.path.join(save_dir, '{:03d}-{:03d}-{:03d}_noisy.png'.format(idx, repeat, self.epoch))
            Image.fromarray(self.noisy255).convert('RGB').save(save_path)

class OneTeacher_TwoBranch(OneTeacher):

    def train_main_process(self):
        if self.train_opt.mode == 'v4_stdmask':
            with torch.no_grad():
                thres = self.loss_params.thres
                ws = self.loss_params.ws if 'ws' in self.loss_params else 7
                self.student_exp = self.backbone(self.input)
                self.teacher1 = self.net_teacher1(self.input)
                teacher1_noise = self.input - self.teacher1
                teacher1_mean = torch.mean(self.teacher1, dim=1, keepdim=True) * 255.0
                std_mask = self.generate_std_mask(teacher1_mean, thres, ws)
                n, c, h, w = self.input.shape
                an_list = teacher1_noise.chunk(n, dim=0)
                mask_list = std_mask.chunk(n, dim=0)
                if n > 1:
                    shift_num = n // 2
                    an_shift_list = an_list[shift_num:] + an_list[:shift_num]
                    an_mask_list = mask_list[shift_num:] + mask_list[:shift_num]
                else:
                    an_shift_list = an_list
                    an_mask_list = mask_list
                teacher1_noise_ranbatch = torch.cat(an_shift_list, dim=0)
                an_mask = torch.cat(an_mask_list, dim=0)
                renoisy = teacher1_noise_ranbatch + self.teacher1
                input_mask = an_mask
            self.high_mask = input_mask
            noiser_mixer = renoisy * input_mask + self.input * (1 - input_mask)

            self.student = self.backbone(noiser_mixer)
            self.teacher_out = self.teacher1
        elif self.train_opt.mode == 'v4_stdmask_poinorm':
            with torch.no_grad():
                thres = self.loss_params.thres
                ws = self.loss_params.ws if 'ws' in self.loss_params else 7
                self.student_exp = self.backbone(self.input)
                self.teacher1 = self.net_teacher1(self.input)
                teacher1_noise = self.input - self.teacher1
                teacher1_noise = teacher1_noise / self.teacher1.clamp(min=0).sqrt().clamp(min=0.001)
                teacher1_mean = torch.mean(self.teacher1, dim=1, keepdim=True) * 255.0
                std_mask = self.generate_std_mask(teacher1_mean, thres, ws)
                n, c, h, w = self.input.shape
                an_list = teacher1_noise.chunk(n, dim=0)
                mask_list = std_mask.chunk(n, dim=0)
                if n > 1:
                    shift_num = n // 2
                    an_shift_list = an_list[shift_num:] + an_list[:shift_num]
                    an_mask_list = mask_list[shift_num:] + mask_list[:shift_num]
                else:
                    an_shift_list = an_list
                    an_mask_list = mask_list
                teacher1_noise_ranbatch = torch.cat(an_shift_list, dim=0)
                an_mask = torch.cat(an_mask_list, dim=0)
                norm_weight = self.loss_params.norm_weight
                renoisy = teacher1_noise_ranbatch * self.teacher1.clamp(min=0).sqrt() * norm_weight + self.teacher1
                input_mask = an_mask
            self.high_mask = input_mask
            noiser_mixer = renoisy * input_mask + self.input * (1 - input_mask)

            self.student = self.backbone(noiser_mixer)
            self.teacher_out = self.teacher1
        elif self.train_opt.mode == 'v4_stdmask_poinorm_v2':
            with torch.no_grad():
                thres = self.loss_params.thres
                ws = self.loss_params.ws if 'ws' in self.loss_params else 7
                norm_bias = self.loss_params.norm_bias
                self.student_exp = self.backbone(self.input)
                self.teacher1 = self.net_teacher1(self.input)
                teacher1_noise = self.input - self.teacher1
                teacher1_noise = teacher1_noise / (self.teacher1.clamp(min=0) + norm_bias).sqrt()
                teacher1_mean = torch.mean(self.teacher1, dim=1, keepdim=True) * 255.0
                std_mask = self.generate_std_mask(teacher1_mean, thres, ws)
                n, c, h, w = self.input.shape
                an_list = teacher1_noise.chunk(n, dim=0)
                mask_list = std_mask.chunk(n, dim=0)
                if n > 1:
                    shift_num = n // 2
                    an_shift_list = an_list[shift_num:] + an_list[:shift_num]
                    an_mask_list = mask_list[shift_num:] + mask_list[:shift_num]
                else:
                    an_shift_list = an_list
                    an_mask_list = mask_list
                teacher1_noise_ranbatch = torch.cat(an_shift_list, dim=0)
                an_mask = torch.cat(an_mask_list, dim=0)
                renoisy = teacher1_noise_ranbatch * (self.teacher1.clamp(min=0) + norm_bias).sqrt() + self.teacher1
                input_mask = an_mask
            self.high_mask = input_mask
            noiser_mixer = renoisy * input_mask + self.input * (1 - input_mask)

            self.student = self.backbone(noiser_mixer)
            self.teacher_out = self.teacher1
        else:
            raise ValueError('Unsupported BSS mode: {}'.format(self.train_opt.mode))

    def train_cal_loss(self):

        difference = (self.student - self.teacher_out) * self.high_mask
        self.loss = torch.sum(difference ** 2) / self.high_mask.sum().clamp(min=1)

class OneTeacher_TwoBranch_SIDDRaw(OneTeacher_TwoBranch, Noise2Clean_SIDDRaw):

    def validate_process_images(self):
        teacher1 = self.teacher1
        teacher1 = self.depth_to_space(teacher1, block_size=2)
        student = self.student
        student = self.depth_to_space(student, block_size=2)
        self.teacher1_01 = teacher1.permute(0, 2, 3, 1).cpu().data.clamp(0, 1).numpy().squeeze(0)
        self.student_01 = student.permute(0, 2, 3, 1).cpu().data.clamp(0, 1).numpy().squeeze(0)
        self.ori_01 = self.data.permute(0, 2, 3, 1).cpu().data.clamp(0, 1).numpy().squeeze(0)
        self.noisy_01 = self.noisy.permute(0, 2, 3, 1).cpu().data.clamp(0, 1).numpy().squeeze(0)
        self.teacher1_255 = tensor2image(teacher1)
        self.student_255 = tensor2image(student)
        self.ori255 = tensor2image(self.data)
        self.noisy255 = tensor2image(self.noisy)
        self.denoise255 = self.student_255

    def get_val_result(self, index, repeat):
        teacher1_psnr = calculate_psnr(self.ori_01.astype(np.float32), self.teacher1_01.astype(np.float32), 1)
        teacher1_ssim = calculate_ssim(self.ori_01.astype(np.float32) * 255, self.teacher1_01.astype(np.float32) * 255)
        student_psnr = calculate_psnr(self.ori_01.astype(np.float32), self.student_01.astype(np.float32), 1)
        student_ssim = calculate_ssim(self.ori_01.astype(np.float32) * 255, self.student_01.astype(np.float32) * 255)
        curr_validate_result = {'img_name': self.data_name, 'teacher1_psnr': teacher1_psnr, 'teacher1_ssim': teacher1_ssim, 'student_psnr': student_psnr, 'student_ssim': student_ssim}
        print(curr_validate_result)
        return curr_validate_result

    def save_results(self, repeat=0, idx=0):
        if self.is_train and self.validate_opt.save_results or (self.is_test and self.test_opt.save_results):
            out = 'val_imgs' if self.is_train else 'test_imgs'
            save_dir = os.path.join(self.train_url, out)
            save_path = os.path.join(save_dir, '{:03d}-{:03d}-{:03d}_dn.MAT'.format(idx, repeat, self.epoch))
            savemat(save_path, {'x': np.ascontiguousarray(self.student_01.squeeze(-1))})
            save_path = os.path.join(save_dir, '{:03d}-{:03d}-{:03d}_clean.MAT'.format(idx, repeat, self.epoch))
            savemat(save_path, {'x': np.ascontiguousarray(self.ori_01.squeeze(-1))})
            save_path = os.path.join(save_dir, '{:03d}-{:03d}-{:03d}_noisy.MAT'.format(idx, repeat, self.epoch))
            savemat(save_path, {'x': np.ascontiguousarray(self.noisy_01.squeeze(-1))})
