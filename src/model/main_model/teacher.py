import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import numpy as np
from PIL import Image
from scipy.io import loadmat,savemat

from src.utils.builder import MASKER,BACKBONE,NOISER,HEAD
from src.utils.img_convert import tensor2image
from src.utils.register import build_from_cfg
from src.metrics.basic import calculate_psnr_from_tensor,calculate_psnr,calculate_ssim

from src.model.main_model.base_model import BaseModel
from src.model.main_model.noise2noise import Noise2Clean,Noise2Clean_SIDDRaw
from src.utils.utils import mkdir


class OneTeacher(Noise2Clean):
    def create_models(self):
        super().create_models()
        self.net_teacher1 = build_from_cfg(self.sub_models_opt.teacher1, BACKBONE)
        # 有些情况需要用这个masker
        if 'masker' in self.sub_models_opt:
            self.train_masker = build_from_cfg(self.sub_models_opt.masker, MASKER)

        self.sub_models.update({
            'teacher1': self.net_teacher1,
        })   


    def std(self,img, window_size=7):
        # print(window_size)
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

    def generate_std_mask(self,input,thres,ws=7):
        N, C, H, W = input.shape
        ratio = input.new_ones((N, 1, H, W)) * 0
        input_std = self.std(input,ws)
        ratio[input_std < thres] = 1
        ratio = ratio.detach()
        return ratio  


    def _unfold_odd(self, x, k):
        assert k % 2 == 1
        pad = k // 2
        x = F.pad(x, [pad] * 4, mode='reflect')
        return F.unfold(x, kernel_size=k)

    def gradient_std_like(self, img01, window_size=7, eps=1e-12):
        """img01: 0-1 gray, returns std-like score (0-255 scale)."""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=img01.dtype, device=img01.device).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3).contiguous()
        gx = F.conv2d(img01, sobel_x, padding=1)
        gy = F.conv2d(img01, sobel_y, padding=1)
        energy = gx * gx + gy * gy
        pad = window_size // 2
        energy = F.avg_pool2d(F.pad(energy, [pad] * 4, mode='reflect'),
                              kernel_size=window_size, stride=1)
        return (255.0 * torch.sqrt(energy.squeeze(1) / 24.0 + eps))

    def laplacian_std_like(self, img01, window_size=7, eps=1e-12):
        """img01: 0-1 gray, returns std-like score (0-255 scale)."""
        lap_k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                             dtype=img01.dtype, device=img01.device).view(1, 1, 3, 3)
        lap = F.conv2d(img01, lap_k, padding=1)
        energy = lap * lap
        pad = window_size // 2
        energy = F.avg_pool2d(F.pad(energy, [pad] * 4, mode='reflect'),
                              kernel_size=window_size, stride=1)
        return (255.0 * torch.sqrt(energy.squeeze(1) / 20.0 + eps))

    def _dct_matrix(self, N, dtype, device):
        n = torch.arange(N, dtype=dtype, device=device)
        k = n.view(-1, 1)
        M = torch.cos(np.pi * (2 * n + 1) * k / (2 * N)) * np.sqrt(2.0 / N)
        M[0] = M[0] / np.sqrt(2.0)
        return M

    def dct_hf_std_like(self, img01, patch_size=8, hf_ratio=0.5, eps=1e-12):
        """img01: 0-1 gray, returns std-like score (0-255 scale)."""
        N, C, H, W = img01.shape
        P = patch_size
        pad_l = (P - 1) // 2
        pad_r = (P - 1) - pad_l
        img_p = F.pad(img01, [pad_l, pad_r, pad_l, pad_r], mode='reflect')
        cols = F.unfold(img_p, kernel_size=P).view(N, C, P, P, H, W)

        M = self._dct_matrix(P, dtype=img01.dtype, device=img01.device)
        coef = torch.einsum('ij,nclkhw->ncjkhw', M, cols)
        coef = torch.einsum('ncikhw,jk->ncijhw', coef, M)

        energy = coef * coef
        energy[:, :, 0, 0] = 0.0

        u = torch.arange(P, device=img01.device).view(P, 1).expand(P, P)
        v = torch.arange(P, device=img01.device).view(1, P).expand(P, P)
        hf_mask = ((u + v).to(img01.dtype) >= hf_ratio * 2 * P).to(img01.dtype)
        hf_mask[0, 0] = 0.0

        hf_energy = (energy * hf_mask.view(1, 1, P, P, 1, 1)).sum(dim=(2, 3))

        num_hf = hf_mask.sum().clamp_min(1.0)
        num_ac = P * P - 1
        rho_hf = num_hf / float(num_ac)

        score = 255.0 * torch.sqrt(hf_energy.squeeze(1) / (rho_hf * P * P + eps))
        return score

    def _norm_per_sample(self, score, p_lo=1, p_hi=99, eps=1e-12):
        n = score.shape[0]
        flat = score.reshape(n, -1)
        lo = torch.quantile(flat, p_lo / 100.0, dim=1, keepdim=True)
        hi = torch.quantile(flat, p_hi / 100.0, dim=1, keepdim=True)
        rng = (hi - lo).clamp(min=eps)
        norm = ((flat - lo) / rng).clamp(0.0, 1.0)
        return norm.view_as(score)

    def generate_diff_mask(self, gray01, kind, thres, **kw):
        """
        gray01: 0-1 gray image [N,1,H,W].
        thres: std-like threshold (e.g. 5).
        Returns: mask where flat=1, textured=0.
        """
        if kind == 'dct':
            score = self.dct_hf_std_like(gray01, kw.get('patch', 8), kw.get('hf_ratio', 0.5))
        elif kind == 'gradient':
            score = self.gradient_std_like(gray01, kw.get('ws', 7))
        elif kind == 'laplacian':
            score = self.laplacian_std_like(gray01, kw.get('ws', 7))
        else:
            raise ValueError('unknown diff mask kind: {}'.format(kind))
        ratio = (score < thres).to(gray01.dtype).unsqueeze(1)
        return ratio.detach()


    def train_main_process(self):
        if self.train_opt.mode == 'shift_batch_std_v4' or self.train_opt.mode == 'shift_batch_std_v3':
            with torch.no_grad():
                thres = self.loss_params.thres

                self.student_exp = self.backbone(self.input)
                self.teacher1 = self.net_teacher1(self.input)
                teacher1_noise = self.input - self.teacher1

                teacher1_mean = torch.mean(self.teacher1,dim=1,keepdim=True)*255.0
                std_mask = self.generate_std_mask(teacher1_mean,thres)

                n,c,h,w = self.input.shape
                an_list = teacher1_noise.chunk(n,dim=0)
                mask_list = std_mask.chunk(n,dim=0)
                if n>1:
                    shift_num = n//2
                    # shift_num = random.randint(1,n-1) 
                    an_shift_list = an_list[shift_num:] + an_list[:shift_num]
                    an_mask_list = mask_list[shift_num:] + mask_list[:shift_num]
                else:
                    an_shift_list = an_list
                    an_mask_list = mask_list
                teacher1_noise_ranbatch = torch.cat(an_shift_list,dim=0)
                an_mask = torch.cat(an_mask_list,dim=0)

            if self.train_opt.mode == 'shift_batch_std_v4':
                out_mask = ((an_mask + std_mask)>0).float()
            elif self.train_opt.mode == 'shift_batch_std_v3':
                out_mask = an_mask
                
            self.student = self.backbone(teacher1_noise_ranbatch*an_mask+self.teacher1 + teacher1_noise*(1-an_mask))*out_mask
            self.avail_mask = out_mask
            self.teacher_out = self.teacher1

        elif self.train_opt.mode == 'std_v4_masker':
            with torch.no_grad():
                thres = self.loss_params.thres

                self.student_exp = self.backbone(self.input)
                self.teacher1 = self.net_teacher1(self.input)
                teacher1_noise = self.input - self.teacher1

                teacher1_mean = torch.mean(self.teacher1,dim=1,keepdim=True)*255.0
                std_mask = self.generate_std_mask(teacher1_mean,thres)

                # 修正std_mask
                _, mask_vol = self.train_masker.add_mask(self.input)
                num_mask = mask_vol.size(0)
                # print(mask_vol.shape)
                # print(std_mask.shape)
                select_num = random.randint(0,num_mask-1)
                # print(select_num)
                select_mask = mask_vol[select_num,...].unsqueeze(0)
                # print(select_mask[:,:,:8,:8])
                std_mask_ = std_mask*select_mask

                n,c,h,w = self.input.shape
                an_list = teacher1_noise.chunk(n,dim=0)
                mask_list = std_mask_.chunk(n,dim=0)

                if n>1:
                    shift_num = n//2
                    an_shift_list = an_list[shift_num:] + an_list[:shift_num]
                    an_mask_list = mask_list[shift_num:] + mask_list[:shift_num]
                else:
                    an_shift_list = an_list
                    an_mask_list = mask_list
                teacher1_noise_ranbatch = torch.cat(an_shift_list,dim=0)
                an_mask = torch.cat(an_mask_list,dim=0)

            out_mask = ((an_mask + std_mask)>0).float()
            self.student = self.backbone(teacher1_noise_ranbatch*an_mask+self.teacher1 + teacher1_noise*(1-an_mask))*out_mask
            self.avail_mask = out_mask
            self.teacher_out = self.teacher1


        elif self.train_opt.mode == 'std_v3_masker':
            with torch.no_grad():
                thres = self.loss_params.thres

                self.student_exp = self.backbone(self.input)
                self.teacher1 = self.net_teacher1(self.input)
                teacher1_noise = self.input - self.teacher1

                teacher1_mean = torch.mean(self.teacher1,dim=1,keepdim=True)*255.0
                std_mask = self.generate_std_mask(teacher1_mean,thres)

                # 修正std_mask
                _, mask_vol = self.train_masker.add_mask(self.input)
                num_mask = mask_vol.size(1)
                # print(mask_vol.shape)
                # print(std_mask.shape)
                select_num = random.randint(0,num_mask-1)
                select_mask = mask_vol[select_num,...].unsqueeze(0)
                std_mask_ = std_mask*select_mask

                n,c,h,w = self.input.shape
                an_list = teacher1_noise.chunk(n,dim=0)
                mask_list = std_mask_.chunk(n,dim=0)

                if n>1:
                    shift_num = n//2
                    an_shift_list = an_list[shift_num:] + an_list[:shift_num]
                    an_mask_list = mask_list[shift_num:] + mask_list[:shift_num]
                else:
                    an_shift_list = an_list
                    an_mask_list = mask_list
                teacher1_noise_ranbatch = torch.cat(an_shift_list,dim=0)
                an_mask = torch.cat(an_mask_list,dim=0)

            out_mask = an_mask
            self.student = self.backbone(teacher1_noise_ranbatch*an_mask+self.teacher1 + teacher1_noise*(1-an_mask))*out_mask
            self.avail_mask = out_mask
            self.teacher_out = self.teacher1

        elif self.train_opt.mode == 'shift_batch':
            with torch.no_grad():
                self.student_exp = self.backbone(self.input)
                self.teacher1 = self.net_teacher1(self.input)
                teacher1_noise = self.input - self.teacher1

                n, c, h, w = self.input.shape
                an_list = teacher1_noise.chunk(n, dim=0)
                if n > 1:
                    shift_num = n // 2
                    an_shift_list = an_list[shift_num:] + an_list[:shift_num]
                else:
                    an_shift_list = an_list
                teacher1_noise_ranbatch = torch.cat(an_shift_list, dim=0)
                renoisy = teacher1_noise_ranbatch + self.teacher1

            self.student = self.backbone(renoisy)
            self.avail_mask = torch.ones_like(self.teacher1[:, :1, :, :])
            self.teacher_out = self.teacher1


    def validate_main_process(self):
        with torch.no_grad():
            self.teacher1 = self.net_teacher1(self.input)
            self.student = self.backbone(self.input)    

    def train_cal_loss(self):
        
        diff = self.student - self.teacher_out
        loss1 = torch.mean(diff**2)
        self.loss = loss1

    def get_msg(self):
        lr = self.scheduler.get_last_lr()[0]

        time_start = self.time_start
        time_end = time.time()
        ori = self.data
        teacher1 = calculate_psnr_from_tensor(ori,self.teacher1)
        student = calculate_psnr_from_tensor(ori,self.student)
        student_exp = calculate_psnr_from_tensor(ori,self.student_exp)
        self.msg = '{:04d} {:05d} lr={:.2e} teacher1={:.6f}, student={:.6f}, student_exp={:.6f}, Loss_All={:.6f}, Time={:.4f}'.format(
                self.epoch,self.curr_iter, lr, teacher1, student, student_exp, self.loss.item(), time_end -time_start)

    def validate_process_images(self):
        # unpadding
        teacher1 = self.teacher1[:,:,:self.H,:self.W]
        student = self.student[:,:,:self.H,:self.W]

        self.teacher1_255 = tensor2image(teacher1)
        self.student_255 = tensor2image(student)
        self.ori255 = tensor2image(self.data)
        self.noisy255 = tensor2image(self.noisy[:,:,:self.H,:self.W])
        self.denoise255 = self.student_255


    def get_val_result(self,repeat,index):
        # calculate metrics
        teacher1_psnr = calculate_psnr(self.ori255,self.teacher1_255)
        teacher1_ssim = calculate_ssim(self.ori255,self.teacher1_255)

        student_psnr = calculate_psnr(self.ori255,self.student_255)
        student_ssim = calculate_ssim(self.ori255,self.student_255)
        data_name = "{:03d}-{:03d}-{:03d}_{}".format(
                    index,repeat, self.epoch, self.data_name[0])
        # collect and append result
        curr_validate_result = {'img_name':data_name,
                                      'teacher1_psnr': teacher1_psnr,
                                      'teacher1_ssim': teacher1_ssim,
                                      'student_psnr': student_psnr,
                                      'student_ssim': student_ssim,
                                      }
        # print(curr_validate_result)
        return curr_validate_result
    
    def logger_val_summary(self,avg_results):
        self.logger.info("epoch:{},teacher1:{:.6f}/{:.6f}, student:{:.6f}/{:.6f}".format(
                    self.epoch, avg_results['teacher1_psnr'], avg_results['teacher1_ssim'],
                                avg_results['student_psnr'], avg_results['student_ssim'],
                                ))

    def save_results(self,repeat=0,idx=0):
        if (self.is_train and self.validate_opt.save_results) or (self.is_test and self.test_opt.save_results):
            out = 'val_imgs' if self.is_train else 'test_imgs'

            opt_testOrVal = self.datasets_opt.validate if self.is_train else self.datasets_opt.test
            name_dataset = opt_testOrVal.data_dir.split('/')[-1]
            save_dir = os.path.join(self.train_url,out,name_dataset)
            mkdir(save_dir)

            save_path = os.path.join(
                save_dir,
                "{:03d}-{:03d}-{:03d}_teacher.png".format(
                    idx,repeat, self.epoch))
            Image.fromarray(self.teacher1_255).convert('RGB').save(save_path)                
            save_path = os.path.join(
                save_dir,
                "{:03d}-{:03d}-{:03d}_student.png".format(
                    idx,repeat, self.epoch))
            Image.fromarray(self.student_255).convert('RGB').save(save_path)                
            # save_path = os.path.join(
            #     save_dir,
            #     "{:03d}-{:03d}-{:03d}_clean.png".format(
            #         idx,repeat, self.epoch))
            # Image.fromarray(self.ori255).convert('RGB').save(save_path)  

            save_path = os.path.join(
                save_dir,
                "{:03d}-{:03d}-{:03d}_noisy.png".format(
                    idx,repeat, self.epoch))
            Image.fromarray(self.noisy255).convert('RGB').save(save_path) 

class OneTeacher_MaksWBalance(OneTeacher):
    def train_cal_loss(self):
        diff = self.student - self.teacher_out
        loss1 = torch.sum(diff**2)
        self.loss = loss1/torch.sum(self.avail_mask)

        # print(self.avail_mask.sum()/torch.ones_like(self.avail_mask).sum())

class OneTeacher_NoiseEstimation(OneTeacher):
    def train_main_process(self):
        if self.train_opt.mode == 'v3_add_truenoise':
            with torch.no_grad():
                thres = self.loss_params.thres
                self.student_exp = self.backbone(self.input)
                self.teacher1 = self.net_teacher1(self.input)
                # teacher1_noise = self.input - self.teacher1

                teacher1_mean = torch.mean(self.teacher1,dim=1,keepdim=True)*255.0
                std_mask = self.generate_std_mask(teacher1_mean,thres)

                n,c,h,w = self.input.shape
                # an_list = teacher1_noise.chunk(n,dim=0)
                mask_list = std_mask.chunk(n,dim=0)
                if n>1:
                    shift_num = n//2
                    # an_shift_list = an_list[shift_num:] + an_list[:shift_num]
                    an_mask_list = mask_list[shift_num:] + mask_list[:shift_num]
                else:
                    # an_shift_list = an_list
                    an_mask_list = mask_list
                # teacher1_noise_ranbatch = torch.cat(an_shift_list,dim=0)
                an_mask = torch.cat(an_mask_list,dim=0)
                renoisy =  self.noise_adder.add_noise(self.teacher1)
            # out_mask = ((an_mask + std_mask)>0).float()
            self.student = self.backbone(renoisy*an_mask+ self.input*(1-an_mask))
            out_mask = an_mask
            self.avail_mask = out_mask
            self.teacher_out = self.teacher1

class OneTeacher_NoiseEstimation_MaksWBalance(OneTeacher_NoiseEstimation):
    def train_cal_loss(self):
        diff = (self.student - self.teacher_out)*self.avail_mask
        loss1 = torch.sum(diff**2)
        self.loss = loss1/torch.sum(self.avail_mask)



class OneTeacher_LowTrainHigh(OneTeacher):
    def train_main_process(self):
        if self.train_opt.mode == 'v4_stdmask':
            with torch.no_grad():
                thres = self.loss_params.thres
                ws = self.loss_params.ws if 'ws' in self.loss_params else 7
                self.student_exp = self.backbone(self.input)
                self.teacher1 = self.net_teacher1(self.input)
                teacher1_noise = self.input - self.teacher1

                teacher1_mean = torch.mean(self.teacher1,dim=1,keepdim=True)*255.0
                std_mask = self.generate_std_mask(teacher1_mean,thres,ws)
                rev_std_mask = 1-std_mask

                n,c,h,w = self.input.shape
                an_list = teacher1_noise.chunk(n,dim=0)
                mask_list = std_mask.chunk(n,dim=0)
                if n>1:
                    shift_num = n//2
                    an_shift_list = an_list[shift_num:] + an_list[:shift_num]
                    an_mask_list = mask_list[shift_num:] + mask_list[:shift_num]
                else:
                    an_shift_list = an_list
                    an_mask_list = mask_list
                teacher1_noise_ranbatch = torch.cat(an_shift_list,dim=0)
                an_mask = torch.cat(an_mask_list,dim=0)
                # renoisy =  self.noise_adder.add_noise(self.teacher1)
                renoisy = teacher1_noise_ranbatch + self.teacher1

                input_mask = an_mask
            self.high_mask = input_mask
            self.low_mask = std_mask
            # out_mask = ((an_mask + std_mask)>0).float()
            self.student = self.backbone(renoisy*input_mask+ self.input*(1-input_mask))
            # out_mask = an_mask
            self.teacher_out = self.teacher1

        elif self.train_opt.mode == 'v5_stdmask':
            with torch.no_grad():
                thres = self.loss_params.thres
                ws = self.loss_params.ws if 'ws' in self.loss_params else 7
                self.student_exp = self.backbone(self.input)
                self.teacher1 = self.net_teacher1(self.input)
                teacher1_noise = self.input - self.teacher1

                teacher1_mean = torch.mean(self.teacher1,dim=1,keepdim=True)*255.0
                std_mask = self.generate_std_mask(teacher1_mean,thres,ws)
                rev_std_mask = 1-std_mask

                n,c,h,w = self.input.shape
                an_list = teacher1_noise.chunk(n,dim=0)
                mask_list = std_mask.chunk(n,dim=0)
                if n>1:
                    shift_num = n//2
                    an_shift_list = an_list[shift_num:] + an_list[:shift_num]
                    an_mask_list = mask_list[shift_num:] + mask_list[:shift_num]
                else:
                    an_shift_list = an_list
                    an_mask_list = mask_list
                teacher1_noise_ranbatch = torch.cat(an_shift_list,dim=0)
                an_mask = torch.cat(an_mask_list,dim=0)
                # renoisy =  self.noise_adder.add_noise(self.teacher1)
                renoisy = teacher1_noise_ranbatch + self.teacher1

                input_mask = rev_std_mask*an_mask
            self.high_mask = input_mask
            self.low_mask = std_mask
            # out_mask = ((an_mask + std_mask)>0).float()
            self.student = self.backbone(renoisy*input_mask+ self.input*(1-input_mask))
            # out_mask = an_mask
            self.teacher_out = self.teacher1

        elif self.train_opt.mode == 'v5_add_truenoise':
            with torch.no_grad():
                thres = self.loss_params.thres
                self.student_exp = self.backbone(self.input)
                self.teacher1 = self.net_teacher1(self.input)
                teacher1_noise = self.input - self.teacher1

                teacher1_mean = torch.mean(self.teacher1,dim=1,keepdim=True)*255.0
                std_mask = self.generate_std_mask(teacher1_mean,thres)
                rev_std_mask = 1-std_mask

                n,c,h,w = self.input.shape
                an_list = teacher1_noise.chunk(n,dim=0)
                mask_list = std_mask.chunk(n,dim=0)
                if n>1:
                    shift_num = n//2
                    an_shift_list = an_list[shift_num:] + an_list[:shift_num]
                    an_mask_list = mask_list[shift_num:] + mask_list[:shift_num]
                else:
                    an_shift_list = an_list
                    an_mask_list = mask_list
                teacher1_noise_ranbatch = torch.cat(an_shift_list,dim=0)
                an_mask = torch.cat(an_mask_list,dim=0)
                renoisy =  self.noise_adder.add_noise(self.teacher1)
                # renoisy = teacher1_noise_ranbatch + self.teacher1

                input_mask = 1-std_mask
            self.high_mask = input_mask
            self.low_mask = std_mask
            # out_mask = ((an_mask + std_mask)>0).float()
            self.student = self.backbone(renoisy*input_mask+ self.input*(1-input_mask))
            # out_mask = an_mask
            self.teacher_out = self.teacher1

    def train_cal_loss(self):
        diff = (self.student - self.teacher_out)
        diff_high = diff*self.high_mask
        diff_low = diff*self.low_mask

        loss_low = torch.sum(diff_low**2)/torch.sum(self.low_mask)
        loss_high = torch.sum(diff_high**2)/torch.sum(self.high_mask)
        low_weight = self.train_opt.low_weight
        self.loss = loss_high + low_weight*loss_low

class OneTeacher_LowTrainHigh_v1_5(OneTeacher_LowTrainHigh):
    def train_cal_loss(self):
        diff = (self.student - self.teacher_out)
        diff_high = diff*self.high_mask
        diff_low = diff*self.low_mask

        loss_low = torch.sum(diff_low**2)/torch.sum(self.low_mask)
        loss_high = torch.sum(diff_high**2)/torch.sum(self.high_mask)
        low_weight = self.train_opt.low_weight
        self.loss = (loss_high + low_weight*loss_low)/(1+low_weight)

        
class OneTeacher_LowTrainHigh_v2(OneTeacher_LowTrainHigh):
    def train_cal_loss(self):
        diff = (self.student - self.teacher_out)
        diff_high = diff*self.high_mask
        diff_low = diff*self.low_mask

        num_low_high = torch.sum(self.low_mask) + torch.sum(self.high_mask)
        loss_low = torch.sum(diff_low**2)/num_low_high
        loss_high = torch.sum(diff_high**2)/num_low_high
        low_weight = self.train_opt.low_weight
        self.loss = loss_high + low_weight*loss_low


class OneTeacher_TwoBranch(OneTeacher):
    def train_main_process(self):
        if self.train_opt.mode == 'v4_stdmask':
            with torch.no_grad():
                thres = self.loss_params.thres
                ws = self.loss_params.ws if 'ws' in self.loss_params else 7
                self.student_exp = self.backbone(self.input)
                self.teacher1 = self.net_teacher1(self.input)
                teacher1_noise = self.input - self.teacher1

                teacher1_mean = torch.mean(self.teacher1,dim=1,keepdim=True)*255.0
                std_mask = self.generate_std_mask(teacher1_mean,thres,ws)
                # rev_std_mask = 1-std_mask

                n,c,h,w = self.input.shape
                an_list = teacher1_noise.chunk(n,dim=0)
                mask_list = std_mask.chunk(n,dim=0)
                if n>1:
                    shift_num = n//2
                    an_shift_list = an_list[shift_num:] + an_list[:shift_num]
                    an_mask_list = mask_list[shift_num:] + mask_list[:shift_num]
                else:
                    an_shift_list = an_list
                    an_mask_list = mask_list
                teacher1_noise_ranbatch = torch.cat(an_shift_list,dim=0)
                an_mask = torch.cat(an_mask_list,dim=0)
                # renoisy =  self.noise_adder.add_noise(self.teacher1)
                renoisy = teacher1_noise_ranbatch + self.teacher1

                input_mask = an_mask
            self.high_mask = input_mask
            self.low_mask = std_mask
            # out_mask = ((an_mask + std_mask)>0).float()
            noiser_mixer = renoisy*input_mask+ self.input*(1-input_mask)
            two_branch_input = torch.cat([noiser_mixer,self.input],dim=0)

            self.two_branch = self.backbone(two_branch_input)
            # out_mask = an_mask
            self.teacher_out = self.teacher1

        elif self.train_opt.mode == 'v4_stdmask_stdnorm':
            # 标准差一致性

            def cal_mask_std(image,mask):
                n,c,h,w = image.shape
                mask = mask.repeat(1,c,1,1)

                # std_list = []
                image = image.view(n,-1)
                mask = mask.view(n,-1)

                mean = torch.sum(image*mask,dim=-1)/ torch.sum(mask,dim=-1).clamp(min=1e-4)
                mean = mean.view(n,1)
                # print(mean.shape)
                # print(image.shape)
                # print(mask.shape)
                var = torch.sum(((image-mean)*mask)**2,dim=-1)/torch.sum(mask,dim=-1).clamp(min=1e-4)
                std = (var.clamp(min=0).sqrt())
                return std


            with torch.no_grad():
                thres = self.loss_params.thres
                ws = self.loss_params.ws if 'ws' in self.loss_params else 7
                self.student_exp = self.backbone(self.input)
                self.teacher1 = self.net_teacher1(self.input)
                teacher1_noise = self.input - self.teacher1

                teacher1_mean = torch.mean(self.teacher1,dim=1,keepdim=True)*255.0
                std_mask = self.generate_std_mask(teacher1_mean,thres,ws)
                # rev_std_mask = 1-std_mask

                n,c,h,w = self.input.shape
                an_list = teacher1_noise.chunk(n,dim=0)
                mask_list = std_mask.chunk(n,dim=0)
                if n>1:
                    shift_num = n//2
                    an_shift_list = an_list[shift_num:] + an_list[:shift_num]
                    an_mask_list = mask_list[shift_num:] + mask_list[:shift_num]
                else:
                    an_shift_list = an_list
                    an_mask_list = mask_list
                teacher1_noise_ranbatch = torch.cat(an_shift_list,dim=0)
                an_mask = torch.cat(an_mask_list,dim=0)
                # renoisy =  self.noise_adder.add_noise(self.teacher1)

                # calculate std ratio
                std_a = cal_mask_std(teacher1_noise_ranbatch,an_mask)
                std_b = cal_mask_std(teacher1_noise,std_mask)
                ratio = std_b/std_a.clamp(min=1e-4)
                # print(std_a,std_b,ratio)

                ratio = ratio.view(-1,1,1,1)
                # print(torch.std((self.input - self.data).view(4,-1),dim=1))


                renoisy = teacher1_noise_ranbatch*ratio + self.teacher1

                input_mask = an_mask
            self.high_mask = input_mask
            self.low_mask = std_mask
            # out_mask = ((an_mask + std_mask)>0).float()
            noiser_mixer = renoisy*input_mask+ self.input*(1-input_mask)
            two_branch_input = torch.cat([noiser_mixer,self.input],dim=0)

            self.two_branch = self.backbone(two_branch_input)
            # out_mask = an_mask
            self.teacher_out = self.teacher1
        elif self.train_opt.mode == 'v4_stdmask_poinorm':
            with torch.no_grad():
                thres = self.loss_params.thres
                ws = self.loss_params.ws if 'ws' in self.loss_params else 7

                self.student_exp = self.backbone(self.input)
                self.teacher1 = self.net_teacher1(self.input)
                teacher1_noise = self.input - self.teacher1
                
                teacher1_noise = teacher1_noise/(self.teacher1.clamp(min=0).sqrt().clamp(min=1e-3))

                teacher1_mean = torch.mean(self.teacher1,dim=1,keepdim=True)*255.0
                std_mask = self.generate_std_mask(teacher1_mean,thres,ws)
                # rev_std_mask = 1-std_mask

                n,c,h,w = self.input.shape
                an_list = teacher1_noise.chunk(n,dim=0)
                mask_list = std_mask.chunk(n,dim=0)
                if n>1:
                    shift_num = n//2
                    an_shift_list = an_list[shift_num:] + an_list[:shift_num]
                    an_mask_list = mask_list[shift_num:] + mask_list[:shift_num]
                else:
                    an_shift_list = an_list
                    an_mask_list = mask_list
                teacher1_noise_ranbatch = torch.cat(an_shift_list,dim=0)
                an_mask = torch.cat(an_mask_list,dim=0)
                # renoisy =  self.noise_adder.add_noise(self.teacher1)
                
                norm_weight = self.loss_params.norm_weight
                renoisy = teacher1_noise_ranbatch*self.teacher1.clamp(min=0).sqrt()*norm_weight + self.teacher1

                input_mask = an_mask
            self.high_mask = input_mask
            self.low_mask = std_mask
            # out_mask = ((an_mask + std_mask)>0).float()
            noiser_mixer = renoisy*input_mask+ self.input*(1-input_mask)
            two_branch_input = torch.cat([noiser_mixer,self.input],dim=0)

            self.two_branch = self.backbone(two_branch_input)
            # out_mask = an_mask
            self.teacher_out = self.teacher1

        elif self.train_opt.mode == 'v4_stdmask_poinorm_v2':
            with torch.no_grad():
                thres = self.loss_params.thres
                ws = self.loss_params.ws if 'ws' in self.loss_params else 7
                norm_bias = self.loss_params.norm_bias

                self.student_exp = self.backbone(self.input)
                self.teacher1 = self.net_teacher1(self.input)
                teacher1_noise = self.input - self.teacher1
                
                teacher1_noise = teacher1_noise/(self.teacher1.clamp(min=0)+ norm_bias).sqrt()

                teacher1_mean = torch.mean(self.teacher1,dim=1,keepdim=True)*255.0
                std_mask = self.generate_std_mask(teacher1_mean,thres,ws)
                # rev_std_mask = 1-std_mask

                n,c,h,w = self.input.shape
                an_list = teacher1_noise.chunk(n,dim=0)
                mask_list = std_mask.chunk(n,dim=0)
                if n>1:
                    shift_num = n//2
                    an_shift_list = an_list[shift_num:] + an_list[:shift_num]
                    an_mask_list = mask_list[shift_num:] + mask_list[:shift_num]
                else:
                    an_shift_list = an_list
                    an_mask_list = mask_list
                teacher1_noise_ranbatch = torch.cat(an_shift_list,dim=0)
                an_mask = torch.cat(an_mask_list,dim=0)
                # renoisy =  self.noise_adder.add_noise(self.teacher1)
                
                renoisy = teacher1_noise_ranbatch*(self.teacher1.clamp(min=0)+norm_bias).sqrt() + self.teacher1

                input_mask = an_mask
            self.high_mask = input_mask
            self.low_mask = std_mask
            # out_mask = ((an_mask + std_mask)>0).float()
            noiser_mixer = renoisy*input_mask+ self.input*(1-input_mask)
            two_branch_input = torch.cat([noiser_mixer,self.input],dim=0)

            if 'clamp' in self.loss_params:
                clamp_range = self.loss_params.clamp
                two_branch_input.clamp(min=clamp_range[0],max=clamp_range[1])

            self.two_branch = self.backbone(two_branch_input)
            # out_mask = an_mask
            self.teacher_out = self.teacher1

        elif self.train_opt.mode == 'v4_stdmask_poinorm_stdnorm':
            def cal_mask_std(image,mask):
                n,c,h,w = image.shape
                mask = mask.repeat(1,c,1,1)

                # std_list = []
                image = image.view(n,-1)
                mask = mask.view(n,-1)

                mean = torch.sum(image*mask,dim=-1)/ torch.sum(mask,dim=-1).clamp(min=1e-4)
                mean = mean.view(n,1)
                # print(mean.shape)
                # print(image.shape)
                # print(mask.shape)
                var = torch.sum(((image-mean)*mask)**2,dim=-1)/torch.sum(mask,dim=-1).clamp(min=1e-4)
                std = (var.clamp(min=0).sqrt())
                return std

            with torch.no_grad():
                thres = self.loss_params.thres
                ws = self.loss_params.ws if 'ws' in self.loss_params else 7
                self.student_exp = self.backbone(self.input)
                self.teacher1 = self.net_teacher1(self.input)
                teacher1_noise = self.input - self.teacher1
                
                teacher1_noise = teacher1_noise/(self.teacher1.clamp(min=0).sqrt().clamp(min=1e-3))

                teacher1_mean = torch.mean(self.teacher1,dim=1,keepdim=True)*255.0
                std_mask = self.generate_std_mask(teacher1_mean,thres,ws)
                # rev_std_mask = 1-std_mask

                n,c,h,w = self.input.shape
                an_list = teacher1_noise.chunk(n,dim=0)
                mask_list = std_mask.chunk(n,dim=0)
                if n>1:
                    shift_num = n//2
                    an_shift_list = an_list[shift_num:] + an_list[:shift_num]
                    an_mask_list = mask_list[shift_num:] + mask_list[:shift_num]
                else:
                    an_shift_list = an_list
                    an_mask_list = mask_list
                teacher1_noise_ranbatch = torch.cat(an_shift_list,dim=0)
                an_mask = torch.cat(an_mask_list,dim=0)

                # calculate std ratio
                std_a = cal_mask_std(teacher1_noise_ranbatch,an_mask)
                std_b = cal_mask_std(teacher1_noise,std_mask)
                ratio = std_b/std_a.clamp(min=1e-4)
                ratio = ratio.view(-1,1,1,1)
                
                norm_weight = self.loss_params.norm_weight
                renoisy = teacher1_noise_ranbatch*ratio*self.teacher1.clamp(min=0).sqrt()*norm_weight + self.teacher1

                input_mask = an_mask
            self.high_mask = input_mask
            self.low_mask = std_mask
            # out_mask = ((an_mask + std_mask)>0).float()
            noiser_mixer = renoisy*input_mask+ self.input*(1-input_mask)
            two_branch_input = torch.cat([noiser_mixer,self.input],dim=0)

            self.two_branch = self.backbone(two_branch_input)
            # out_mask = an_mask
            self.teacher_out = self.teacher1



        elif self.train_opt.mode == 'v4_stdmask_poinorm_stdnorm_v2':
            def cal_mask_std(image,mask):
                n,c,h,w = image.shape
                mask = mask.repeat(1,c,1,1)

                # std_list = []
                image = image.view(n,-1)
                mask = mask.view(n,-1)

                mean = torch.sum(image*mask,dim=-1)/ torch.sum(mask,dim=-1).clamp(min=1e-4)
                mean = mean.view(n,1)
                # print(mean.shape)
                # print(image.shape)
                # print(mask.shape)
                var = torch.sum(((image-mean)*mask)**2,dim=-1)/torch.sum(mask,dim=-1).clamp(min=1e-4)
                std = (var.clamp(min=0).sqrt())
                return std

            with torch.no_grad():
                thres = self.loss_params.thres
                ws = self.loss_params.ws if 'ws' in self.loss_params else 7
                norm_bias = self.loss_params.norm_bias

                self.student_exp = self.backbone(self.input)
                self.teacher1 = self.net_teacher1(self.input)
                teacher1_noise = self.input - self.teacher1
                
                # norm
                teacher1_noise = teacher1_noise/(self.teacher1.clamp(min=0) + norm_bias).sqrt()

                teacher1_mean = torch.mean(self.teacher1,dim=1,keepdim=True)*255.0
                std_mask = self.generate_std_mask(teacher1_mean,thres,ws)
                # rev_std_mask = 1-std_mask

                n,c,h,w = self.input.shape
                an_list = teacher1_noise.chunk(n,dim=0)
                mask_list = std_mask.chunk(n,dim=0)
                if n>1:
                    shift_num = n//2
                    an_shift_list = an_list[shift_num:] + an_list[:shift_num]
                    an_mask_list = mask_list[shift_num:] + mask_list[:shift_num]
                else:
                    an_shift_list = an_list
                    an_mask_list = mask_list
                teacher1_noise_ranbatch = torch.cat(an_shift_list,dim=0)
                an_mask = torch.cat(an_mask_list,dim=0)

                # calculate std ratio
                std_a = cal_mask_std(teacher1_noise_ranbatch,an_mask)
                std_b = cal_mask_std(teacher1_noise,std_mask)
                # print(std_a,std_b)
                ratio = std_b/std_a.clamp(min=1e-4)
                ratio = ratio.view(-1,1,1,1)
                
                # renorm
                renoisy = teacher1_noise_ranbatch*ratio*(self.teacher1.clamp(min=0)+norm_bias).sqrt() + self.teacher1

                input_mask = an_mask
            self.high_mask = input_mask
            self.low_mask = std_mask
            # out_mask = ((an_mask + std_mask)>0).float()
            noiser_mixer = renoisy*input_mask+ self.input*(1-input_mask)
            two_branch_input = torch.cat([noiser_mixer,self.input],dim=0)

            self.two_branch = self.backbone(two_branch_input)
            # out_mask = an_mask
            self.teacher_out = self.teacher1

        elif self.train_opt.mode == 'v4_dctmask':
            with torch.no_grad():
                thres = self.loss_params.thres
                patch = self.loss_params.patch if 'patch' in self.loss_params else 8
                hf_ratio = self.loss_params.hf_ratio if 'hf_ratio' in self.loss_params else 0.5
                self.student_exp = self.backbone(self.input)
                self.teacher1 = self.net_teacher1(self.input)
                teacher1_noise = self.input - self.teacher1

                teacher1_mean = torch.mean(self.teacher1,dim=1,keepdim=True)
                std_mask = self.generate_diff_mask(teacher1_mean, 'dct', thres, patch=patch, hf_ratio=hf_ratio)

                n,c,h,w = self.input.shape
                an_list = teacher1_noise.chunk(n,dim=0)
                mask_list = std_mask.chunk(n,dim=0)
                if n>1:
                    shift_num = n//2
                    an_shift_list = an_list[shift_num:] + an_list[:shift_num]
                    an_mask_list = mask_list[shift_num:] + mask_list[:shift_num]
                else:
                    an_shift_list = an_list
                    an_mask_list = mask_list
                teacher1_noise_ranbatch = torch.cat(an_shift_list,dim=0)
                an_mask = torch.cat(an_mask_list,dim=0)
                renoisy = teacher1_noise_ranbatch + self.teacher1

                input_mask = an_mask
            self.high_mask = input_mask
            self.low_mask = std_mask
            noiser_mixer = renoisy*input_mask+ self.input*(1-input_mask)
            two_branch_input = torch.cat([noiser_mixer,self.input],dim=0)

            self.two_branch = self.backbone(two_branch_input)
            self.teacher_out = self.teacher1

        elif self.train_opt.mode == 'v4_gradmask':
            with torch.no_grad():
                thres = self.loss_params.thres
                ws = self.loss_params.ws if 'ws' in self.loss_params else 7
                self.student_exp = self.backbone(self.input)
                self.teacher1 = self.net_teacher1(self.input)
                teacher1_noise = self.input - self.teacher1

                teacher1_mean = torch.mean(self.teacher1,dim=1,keepdim=True)
                std_mask = self.generate_diff_mask(teacher1_mean, 'gradient', thres, ws=ws)

                n,c,h,w = self.input.shape
                an_list = teacher1_noise.chunk(n,dim=0)
                mask_list = std_mask.chunk(n,dim=0)
                if n>1:
                    shift_num = n//2
                    an_shift_list = an_list[shift_num:] + an_list[:shift_num]
                    an_mask_list = mask_list[shift_num:] + mask_list[:shift_num]
                else:
                    an_shift_list = an_list
                    an_mask_list = mask_list
                teacher1_noise_ranbatch = torch.cat(an_shift_list,dim=0)
                an_mask = torch.cat(an_mask_list,dim=0)
                renoisy = teacher1_noise_ranbatch + self.teacher1

                input_mask = an_mask
            self.high_mask = input_mask
            self.low_mask = std_mask
            noiser_mixer = renoisy*input_mask+ self.input*(1-input_mask)
            two_branch_input = torch.cat([noiser_mixer,self.input],dim=0)

            self.two_branch = self.backbone(two_branch_input)
            self.teacher_out = self.teacher1

        elif self.train_opt.mode == 'v4_lapmask':
            with torch.no_grad():
                thres = self.loss_params.thres
                ws = self.loss_params.ws if 'ws' in self.loss_params else 7
                self.student_exp = self.backbone(self.input)
                self.teacher1 = self.net_teacher1(self.input)
                teacher1_noise = self.input - self.teacher1

                teacher1_mean = torch.mean(self.teacher1,dim=1,keepdim=True)
                std_mask = self.generate_diff_mask(teacher1_mean, 'laplacian', thres, ws=ws)

                n,c,h,w = self.input.shape
                an_list = teacher1_noise.chunk(n,dim=0)
                mask_list = std_mask.chunk(n,dim=0)
                if n>1:
                    shift_num = n//2
                    an_shift_list = an_list[shift_num:] + an_list[:shift_num]
                    an_mask_list = mask_list[shift_num:] + mask_list[:shift_num]
                else:
                    an_shift_list = an_list
                    an_mask_list = mask_list
                teacher1_noise_ranbatch = torch.cat(an_shift_list,dim=0)
                an_mask = torch.cat(an_mask_list,dim=0)
                renoisy = teacher1_noise_ranbatch + self.teacher1

                input_mask = an_mask
            self.high_mask = input_mask
            self.low_mask = std_mask
            noiser_mixer = renoisy*input_mask+ self.input*(1-input_mask)
            two_branch_input = torch.cat([noiser_mixer,self.input],dim=0)

            self.two_branch = self.backbone(two_branch_input)
            self.teacher_out = self.teacher1



    def train_cal_loss(self):
        out_mixer, out_ori = torch.chunk(self.two_branch,2,dim=0)
        self.student = out_mixer
        diff_mixer = (out_mixer - self.teacher_out)
        diff_high = diff_mixer*self.high_mask

        diff_ori = out_ori - self.teacher_out
        diff_low = diff_ori*self.low_mask

        # print(torch.mean(self.low_mask),torch.mean(self.high_mask))

        num_low_high = torch.sum(self.low_mask) + torch.sum(self.high_mask)
        num_low_high = num_low_high.clamp(min=1) # aviod divide zero
        loss_low = torch.sum(diff_low**2)/num_low_high
        loss_high = torch.sum(diff_high**2)/num_low_high
        low_weight = self.train_opt.low_weight
        # print(loss_high.item())
        self.loss = loss_high + low_weight*loss_low


class OneTeacher_TwoBranch_SIDDRaw(OneTeacher_TwoBranch,Noise2Clean_SIDDRaw):
    def validate_process_images(self):
        teacher1 = self.teacher1
        teacher1 = self.depth_to_space(teacher1,block_size=2)
        student = self.student
        student = self.depth_to_space(student,block_size=2)

        self.teacher1_01 = teacher1.permute(0,2,3,1).cpu().data.clamp(0,1).numpy().squeeze(0)
        self.student_01 = student.permute(0,2,3,1).cpu().data.clamp(0,1).numpy().squeeze(0)
        self.ori_01 = self.data.permute(0,2,3,1).cpu().data.clamp(0,1).numpy().squeeze(0)
        self.noisy_01 = self.noisy.permute(0,2,3,1).cpu().data.clamp(0,1).numpy().squeeze(0)

        self.teacher1_255 = tensor2image(teacher1)
        self.student_255 = tensor2image(student)
        self.ori255 = tensor2image(self.data)
        self.noisy255 = tensor2image(self.noisy)
        self.denoise255 = self.student_255

    def get_val_result(self,index,repeat):
        # calculate metrics
        teacher1_psnr = calculate_psnr(self.ori_01.astype(np.float32),self.teacher1_01.astype(np.float32),1)
        teacher1_ssim = calculate_ssim(self.ori_01.astype(np.float32)*255,self.teacher1_01.astype(np.float32)*255)

        student_psnr = calculate_psnr(self.ori_01.astype(np.float32),self.student_01.astype(np.float32),1)
        student_ssim = calculate_ssim(self.ori_01.astype(np.float32)*255,self.student_01.astype(np.float32)*255)
        # student_psnr = calculate_psnr(self.ori255,self.student_255)
        # student_ssim = calculate_ssim(self.ori255,self.student_255)
        # collect and append result
        curr_validate_result = {'img_name':self.data_name,
                                      'teacher1_psnr': teacher1_psnr,
                                      'teacher1_ssim': teacher1_ssim,
                                      'student_psnr': student_psnr,
                                      'student_ssim': student_ssim,
                                      }
        print(curr_validate_result)
        return curr_validate_result

    def save_results(self,repeat=0,idx=0):
        if (self.is_train and self.validate_opt.save_results) or (self.is_test and self.test_opt.save_results):
            out = 'val_imgs' if self.is_train else 'test_imgs'
            save_dir = os.path.join(self.train_url,out)

            save_path = os.path.join(
                save_dir,
                "{:03d}-{:03d}-{:03d}_dn.MAT".format(
                    idx,repeat, self.epoch))

            savemat(save_path, {"x": np.ascontiguousarray(self.student_01.squeeze(-1))})

            save_path = os.path.join(
                save_dir,
                "{:03d}-{:03d}-{:03d}_clean.MAT".format(
                    idx,repeat, self.epoch))

            savemat(save_path, {"x": np.ascontiguousarray(self.ori_01.squeeze(-1))})

            save_path = os.path.join(
                save_dir,
                "{:03d}-{:03d}-{:03d}_noisy.MAT".format(
                    idx,repeat, self.epoch))

            savemat(save_path, {"x": np.ascontiguousarray(self.noisy_01.squeeze(-1))})

class OneTeacher_TwoBranch_Fixed(OneTeacher_TwoBranch):
    # 之前把mask当作一通道了。。
    def train_cal_loss(self):
        out_mixer, out_ori = torch.chunk(self.two_branch,2,dim=0)
        self.high_mask = self.high_mask.repeat(1,3,1,1)
        self.low_mask = self.low_mask.repeat(1,3,1,1)
        self.student = out_mixer
        diff_mixer = (out_mixer - self.teacher_out)

        diff_high = diff_mixer*self.high_mask

        diff_ori = out_ori - self.teacher_out
        diff_low = diff_ori*self.low_mask

        num_low_high = torch.sum(self.low_mask) + torch.sum(self.high_mask)
        loss_low = torch.sum(diff_low**2)/num_low_high
        loss_high = torch.sum(diff_high**2)/num_low_high
        low_weight = self.train_opt.low_weight
        self.loss = loss_high + low_weight*loss_low
