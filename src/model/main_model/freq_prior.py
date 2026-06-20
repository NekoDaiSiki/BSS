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
from src.utils.utils import mkdir

from .blind_spot import B2UB_Model

class Frequency_B2UB_Model(B2UB_Model):
    def create_models(self):
        super().create_models()
        self.anchor = build_from_cfg(self.sub_models_opt.anchor,BACKBONE)
        self.sub_models.update({
            'anchor': self.anchor,
        })
    
    # def resume(self):
    #     super().resume()
        # copy backbone parms to anchor if resume path is not given
        
        # if self.resume_opt.path and self.resume_opt.path.anchor:
        #     for param_q, param_k in zip(
        #         self.backbone.parameters(), self.anchor.parameters()
        #     ):
        #         param_k.data = param_q.data
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

    def generate_soft_multiplier(self,input,lower=1,mullower=1,upper=30,mulupper=0,ws=7):
        # print(ws)
        N, C, H, W = input.shape
        ratio = input.new_ones((N, 1, H, W)) * 1
        input_std = self.std(input, ws)
        # print(torch.mean((input_std<0.001).float()))
        ratio[input_std < lower] = (mullower-1)*2*(0.5-torch.sigmoid((input_std - lower)))[input_std < lower] + 1
        ratio[input_std > upper] = (1-mulupper)*2*(0.5-torch.sigmoid((input_std - upper)))[input_std > upper] + 1
        ratio = ratio.detach()
        return ratio

    def generate_upper(self,input,upper=5,ws=7):
        N, C, H, W = input.shape
        ratio = input.new_ones((N, 1, H, W)) * 0
        input_std = self.std(input,ws)
        ratio[input_std > upper] = 1
        ratio = ratio.detach()
        return ratio

    def gradient_std_like(self, img01, window_size=7, eps=1e-12):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=img01.dtype, device=img01.device).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3).contiguous()
        gx = F.conv2d(img01, sobel_x, padding=1)
        gy = F.conv2d(img01, sobel_y, padding=1)
        energy = gx * gx + gy * gy
        pad = window_size // 2
        energy = F.avg_pool2d(F.pad(energy, [pad] * 4, mode='reflect'),
                              kernel_size=window_size, stride=1)
        return 255.0 * torch.sqrt(energy.squeeze(1) / 24.0 + eps)

    def laplacian_std_like(self, img01, window_size=7, eps=1e-12):
        lap_k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                             dtype=img01.dtype, device=img01.device).view(1, 1, 3, 3)
        lap = F.conv2d(img01, lap_k, padding=1)
        energy = lap * lap
        pad = window_size // 2
        energy = F.avg_pool2d(F.pad(energy, [pad] * 4, mode='reflect'),
                              kernel_size=window_size, stride=1)
        return 255.0 * torch.sqrt(energy.squeeze(1) / 20.0 + eps)

    def generate_gradient_soft_flat_mask(self, img01, thres=5, ws=7, sharpness=1.0):
        score = self.gradient_std_like(img01, window_size=ws)
        mask = torch.sigmoid(sharpness * (thres - score)).unsqueeze(1)
        return mask.detach()

    def generate_laplacian_soft_flat_mask(self, img01, thres=2, ws=7, sharpness=1.0):
        score = self.laplacian_std_like(img01, window_size=ws)
        mask = torch.sigmoid(sharpness * (thres - score)).unsqueeze(1)
        return mask.detach()

    def _dct_matrix(self, P, dtype, device):
        n = torch.arange(P, dtype=dtype, device=device)
        k = n.view(-1, 1)
        M = torch.cos(np.pi * (2 * n + 1) * k / (2 * P)) * np.sqrt(2.0 / P)
        M[0] = M[0] / np.sqrt(2.0)
        return M

    def dct_hf_std_like(self, img01, patch_size=8, hf_ratio=0.5, eps=1e-12):
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

        return 255.0 * torch.sqrt(hf_energy.squeeze(1) / (rho_hf * P * P + eps))

    def generate_dct_soft_flat_mask(self, img01, thres=2, patch=2, hf_ratio=0.5, sharpness=1.0):
        score = self.dct_hf_std_like(img01, patch_size=patch, hf_ratio=hf_ratio)
        mask = torch.sigmoid(sharpness * (thres - score)).unsqueeze(1)
        return mask.detach()


    def train_cal_loss(self):
        diff = self.denoise - self.target
        exp_diff = (self.exp_denoise - self.target)* self.denoise
        alpha,beta = self.get_beta()
        revisible = exp_diff
        self.beta = beta
        self.loss_reg = alpha * torch.mean(diff**2)
        self.loss_rev = beta*torch.mean(revisible)

        loss_mode = self.train_opt.loss_params.mode
        if  loss_mode == 'base':
            pass
        elif loss_mode == 'freq_reg_soft_multiplier':
            lower = self.loss_params.lower
            mullower = self.loss_params.mullower
            upper = self.loss_params.upper
            mulupper = self.loss_params.mulupper
            ws = self.loss_params.ws if self.loss_params.ws else 7
            exp_mean = torch.mean(self.exp_denoise,dim=1,keepdim=True)*255.0
            freq_mask = self.generate_soft_multiplier(exp_mean,lower=lower,mullower=mullower,upper=upper,mulupper=mulupper,ws=ws)
            # print(upper)
            # print(torch.mean(freq_mask))
            self.loss_reg = alpha * torch.mean(freq_mask*(diff**2))

        self.loss = self.loss_reg + self.loss_rev


    def get_msg(self):
        lr = self.scheduler.get_last_lr()[0]

        time_start = self.time_start
        time_end = time.time()

        dn = self.denoise
        dn_exp = self.exp_denoise
        ori = self.data
        distortion = calculate_psnr_from_tensor(ori,dn)
        distortion_exp = calculate_psnr_from_tensor(ori,dn_exp)
        anchor_exp = calculate_psnr_from_tensor(ori,self.anchor_exp)
        self.msg = '{:04d} {:05d} lr={:.2e} distortion={:.6f}, distortion_exp={:.6f}, anchor_exp={:.6f},Loss_Reg={:.6f}, Beta={}, Loss_Rev={:.6f}, Loss_All={:.6f}, Time={:.4f}'.format(
                self.epoch,self.curr_iter, lr, distortion, distortion_exp ,anchor_exp, \
                        self.loss_reg.item(), self.beta, self.loss_rev.item(), self.loss.item(), time_end -time_start)
    
    #  for fast testing
    def validate_init(self):
        self.data = self.data/255.0
        # add noise
        self.noisy = self.train_noise_adder.add_noise(self.data)
        # padding to square, need to research it
        n,c,h,w = self.data.shape
        self.H,self.W = h,w
        val_size = (max(h, w) + 31) // 32 * 32
        self.noisy = F.pad(self.noisy,[0,val_size-w,0,val_size-h],mode='reflect')
        # self.input, self.mask = self.train_masker.add_mask(self.noisy)

    def validate_main_process(self):
        with torch.no_grad():
            self.exp_denoise = self.backbone(self.noisy)
    
    def validate_process_images(self):
        # unpadding
        denoise_exp = self.exp_denoise[:,:,:self.H,:self.W]
        self.denoise255_exp = tensor2image(denoise_exp)
        self.ori255 = tensor2image(self.data)
        self.noisy255 = tensor2image(self.noisy[:,:,:self.H,:self.W])

    def get_val_result(self,repeat,index):
        # calculate metrics
        denoise_exp_psnr = calculate_psnr(self.ori255,self.denoise255_exp)
        denoise_exp_ssim = calculate_ssim(self.ori255,self.denoise255_exp)
        # collect and append result
        data_name = "{:03d}-{:03d}-{:03d}_{}".format(
                    index,repeat, self.epoch, self.data_name[0])
        curr_validate_result={'img_name':data_name,
                                      'denoise_exp_psnr': denoise_exp_psnr,
                                      'denoise_exp_ssim': denoise_exp_ssim,
                                      }
        # self.logger.info(curr_validate_result)
        return curr_validate_result
    def save_results(self,repeat=0,idx=0):
        if (self.is_train and self.validate_opt.save_results) or (self.is_test and self.test_opt.save_results):
            out = 'val_imgs' if self.is_train else 'test_imgs'
            
            opt_testOrVal = self.datasets_opt.validate if self.is_train else self.datasets_opt.test
            name_dataset = opt_testOrVal.data_dir.split('/')[-1]
            save_dir = os.path.join(self.train_url,out,name_dataset)
            mkdir(save_dir)


            save_path = os.path.join(
                save_dir,
                "{:03d}-{:03d}-{:03d}_exp.png".format(
                    idx,repeat,  self.epoch))
            Image.fromarray(self.denoise255_exp).convert('RGB').save(save_path)
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


    def logger_val_summary(self,avg_results):
        self.logger.info("epoch:{},exp:{:.6f}/{:.6f}".format(
                    self.epoch,
                        avg_results['denoise_exp_psnr'], avg_results['denoise_exp_ssim'], 
                        ))

class Frequency_B2UB_Model_Momentum(Frequency_B2UB_Model):
    def __init__(self, opt):
        super().__init__(opt)
        self.m = self.main_model_opt.m

    @torch.no_grad()
    def _momentum_update(self,encoder_q,encoder_k):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            encoder_q.parameters(), encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)    


    def train_main_process(self):
        self._momentum_update(self.backbone,self.anchor)
        super().train_main_process()   

    def train_cal_loss(self):
        diff = self.denoise - self.target
        anchor_diff = (self.anchor_exp - self.target)* self.denoise
        alpha,beta = self.get_beta()
        revisible = anchor_diff
        self.beta = beta
        self.loss_reg = alpha * torch.mean(diff**2)
        self.loss_rev = beta*torch.mean(revisible)

        loss_mode = self.train_opt.loss_params.mode
        if  loss_mode == 'base':
            pass
        elif loss_mode == 'freq_reg_soft_multiplier':
            lower = self.loss_params.lower
            mullower = self.loss_params.mullower
            upper = self.loss_params.upper
            mulupper = self.loss_params.mulupper
            exp_mean = torch.mean(self.exp_denoise,dim=1,keepdim=True)*255.0
            freq_mask = self.generate_soft_multiplier(exp_mean,lower=lower,mullower=mullower,upper=upper,mulupper=mulupper)
            self.loss_reg = alpha * torch.mean(freq_mask*(diff**2))

        self.loss = self.loss_reg + self.loss_rev
    def validate_main_process(self):
        super().validate_main_process()
        with torch.no_grad():
            self.anchor_exp = self.anchor(self.noisy)
    
    def validate_process_images(self):
        super().validate_process_images()
        anchor_exp = self.anchor_exp[:,:,:self.H,:self.W]
        self.anchor255_exp = tensor2image(anchor_exp)

    def get_val_result(self,repeat,index):
        curr_validate_result = super().get_val_result(repeat,index)
        anchor_exp_psnr = calculate_psnr(self.ori255,self.anchor255_exp)
        anchor_exp_ssim = calculate_ssim(self.ori255,self.anchor255_exp)

        curr_validate_result.update({'anchor_exp_psnr':anchor_exp_psnr,
                                    'anchor_exp_ssim': anchor_exp_ssim})
        # print(curr_validate_result)
        return curr_validate_result

    def logger_val_summary(self,avg_results):
        self.logger.info("epoch:{},dn:{:.6f}/{:.6f},exp:{:.6f}/{:.6f},anchor:{:.6f}/{:.6f},mid:{:.6f}/{:.6f}".format(
                    self.epoch, avg_results['denoise_psnr'], avg_results['denoise_ssim'], 
                        avg_results['denoise_exp_psnr'], avg_results['denoise_exp_ssim'], 
                        avg_results['anchor_exp_psnr'], avg_results['anchor_exp_ssim'], 
                        avg_results['denoise_mid_psnr'], avg_results['denoise_mid_ssim']))


    def save_results(self, repeat=0, idx=0):
        super().save_results(repeat, idx)
        if (self.is_train and self.validate_opt.save_results) or (self.is_test and self.test_opt.save_results):
            out = 'val_imgs' if self.is_train else 'test_imgs'
            save_dir = os.path.join(self.train_url,out)

            save_path = os.path.join(
                save_dir,
                "{:03d}-{:03d}-{:03d}_anchor.png".format(
                    idx,repeat,  self.epoch))
            Image.fromarray(self.anchor255_exp).convert('RGB').save(save_path)


class Frequency_B2UB_Model_SupStd(Frequency_B2UB_Model):
    def meanPerChannel(self,img, window_size=3):
        assert window_size % 2 == 1
        pad = window_size // 2
        # calculate std on the mean image of the color channels
        N, C, H, W = img.shape
        img = nn.functional.pad(img, [pad] * 4, mode='reflect')
        img = nn.functional.unfold(img, kernel_size=window_size)
        img = img.view(N, C, window_size * window_size, H, W)
        
        if self.train_opt.clear_center:
            # print('activete')
            img = torch.cat([img[:,:,:window_size**2//2,...],img[:,:,window_size**2//2 +1:,...]],dim=2)
        # img = img - torch.mean(img, dim=2, keepdim=True)
        img_mean = torch.mean(img, dim=2) # N,C,H,W
        # img = img * img
        # img = torch.mean(img, dim=2, keepdim=True)
        # img = torch.sqrt(img)
        # img = img.squeeze(2)
        return img_mean
    
    def calMeanLoss(self,direct_dn,exp_dn):
        parms = self.train_opt.loss_params.mean_loss_params
        weight = parms.weight
        ws = parms.ws
        thres = parms.thres

        mean_exp = self.meanPerChannel(exp_dn,ws)
        mean_loss = (direct_dn - mean_exp)**2 

        mask_mode = parms.mode if 'mode' in parms else 'std_hard'
        if mask_mode == 'dct_soft':
            patch = parms.patch if 'patch' in parms else 2
            hf_ratio = parms.hf_ratio if 'hf_ratio' in parms else 0.5
            sharpness = parms.sharpness if 'sharpness' in parms else 1.0
            mask_input = torch.mean(exp_dn,dim=1,keepdim=True)
            mask = self.generate_dct_soft_flat_mask(
                mask_input,
                thres=thres,
                patch=patch,
                hf_ratio=hf_ratio,
                sharpness=sharpness,
            )
        elif mask_mode == 'grad_soft':
            sharpness = parms.sharpness if 'sharpness' in parms else 1.0
            mask_input = torch.mean(exp_dn,dim=1,keepdim=True)
            mask = self.generate_gradient_soft_flat_mask(
                mask_input,
                thres=thres,
                ws=ws,
                sharpness=sharpness,
            )
        elif mask_mode == 'lap_soft':
            sharpness = parms.sharpness if 'sharpness' in parms else 1.0
            mask_input = torch.mean(exp_dn,dim=1,keepdim=True)
            mask = self.generate_laplacian_soft_flat_mask(
                mask_input,
                thres=thres,
                ws=ws,
                sharpness=sharpness,
            )
        else:
            mask_input = torch.mean(exp_dn,dim=1,keepdim=True)*255.0
            mask = 1-self.generate_upper(mask_input,thres,ws)

        return weight*torch.mean(mask*mean_loss)

    def train_main_process(self):
        super().train_main_process()
        self.dirct_denoise = self.backbone(self.noisy)

    def train_cal_loss(self):
        super().train_cal_loss()
        self.loss_mean = self.beta*self.calMeanLoss(self.dirct_denoise,self.exp_denoise)

        self.loss = self.loss_reg + self.loss_rev + self.loss_mean
        # print('father cal loss')

    def get_msg(self):
        lr = self.scheduler.get_last_lr()[0]

        time_start = self.time_start
        time_end = time.time()

        dn = self.denoise
        dn_exp = self.exp_denoise
        ori = self.data
        distortion = calculate_psnr_from_tensor(ori,dn)
        distortion_exp = calculate_psnr_from_tensor(ori,dn_exp)
        anchor_exp = calculate_psnr_from_tensor(ori,self.anchor_exp)

        self.msg = '{:04d} {:05d} lr={:.2e} distortion={:.6f}, distortion_exp={:.6f}, anchor_exp={:.6f},Loss_Reg={:.6f}, Beta={}, Loss_Rev={:.6f}, Loss_Mean={:.6f}, Loss_All={:.6f}, Time={:.4f}'.format(
                self.epoch,self.curr_iter, lr, distortion, distortion_exp ,anchor_exp, \
                        self.loss_reg.item(), self.beta, self.loss_rev.item(),self.loss_mean, self.loss.item(), time_end -time_start)


class Frequency_B2UB_Model_SupStd_Anchor(Frequency_B2UB_Model_SupStd):
    def train_cal_loss(self):
        super(Frequency_B2UB_Model_SupStd,self).train_cal_loss()

        if self.train_opt.loss_params.mean_loss_params.idp_weight:
            # print('idp weight')
            self.loss_mean = self.calMeanLoss(self.dirct_denoise,self.anchor_exp)
        else:
            self.loss_mean = self.beta*self.calMeanLoss(self.dirct_denoise,self.anchor_exp)

        self.loss = self.loss_reg + self.loss_rev + self.loss_mean

class Frequency_B2UB_Model_SupStd_AnchorMomentum(Frequency_B2UB_Model_SupStd_Anchor):
    def __init__(self, opt):
        super().__init__(opt)
        self.m = self.main_model_opt.m

    @torch.no_grad()
    def _momentum_update(self,encoder_q,encoder_k):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            encoder_q.parameters(), encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)    


    def train_main_process(self):
        self._momentum_update(self.backbone,self.anchor)
        super().train_main_process()   

    def validate_main_process(self):
        super().validate_main_process()
        with torch.no_grad():
            self.anchor_exp = self.anchor(self.noisy)
    
    def validate_process_images(self):
        super().validate_process_images()
        anchor_exp = self.anchor_exp[:,:,:self.H,:self.W]
        self.anchor255_exp = tensor2image(anchor_exp)

    def get_val_result(self,repeat,index):
        curr_validate_result = super().get_val_result(repeat,index)
        anchor_exp_psnr = calculate_psnr(self.ori255,self.anchor255_exp)
        anchor_exp_ssim = calculate_ssim(self.ori255,self.anchor255_exp)

        curr_validate_result.update({'anchor_exp_psnr':anchor_exp_psnr,
                                    'anchor_exp_ssim': anchor_exp_ssim})
        print(curr_validate_result)
        return curr_validate_result

    def logger_val_summary(self,avg_results):
        self.logger.info("epoch:{},exp:{:.6f}/{:.6f},anchor:{:.6f}/{:.6f}".format(
                        self.epoch,
                        avg_results['denoise_exp_psnr'], avg_results['denoise_exp_ssim'], 
                        avg_results['anchor_exp_psnr'], avg_results['anchor_exp_ssim'], ))


    def save_results(self, repeat=0, idx=0):
        super().save_results(repeat, idx)
        if (self.is_train and self.validate_opt.save_results) or (self.is_test and self.test_opt.save_results):
            out = 'val_imgs' if self.is_train else 'test_imgs'
            opt_testOrVal = self.datasets_opt.validate if self.is_train else self.datasets_opt.test
            name_dataset = opt_testOrVal.data_dir.split('/')[-1]
            save_dir = os.path.join(self.train_url,out,name_dataset)
            mkdir(save_dir)
            # print(save_dir)

            save_path = os.path.join(
                save_dir,
                "{:03d}-{:03d}-{:03d}_anchor.png".format(
                    idx,repeat,  self.epoch))
            Image.fromarray(self.anchor255_exp).convert('RGB').save(save_path)


