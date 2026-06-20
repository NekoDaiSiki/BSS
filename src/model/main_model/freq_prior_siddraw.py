
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
from .freq_prior import Frequency_B2UB_Model


class Frequency_B2UB_Model_SIDDRaw(Frequency_B2UB_Model):
    def space_to_depth(self, x, block_size):
        n, c, h, w = x.size()
        unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
        return unfolded_x.view(n, c * block_size**2, h // block_size,
                            w // block_size)
    
    def depth_to_space(self,x, block_size):
        return torch.nn.functional.pixel_shuffle(x, block_size)
    
    def train_init(self):
        self.time_start = time.time()
        self.data = self.space_to_depth(self.data,2)
        self.noisy = self.data
        self.input, self.mask = self.train_masker.add_mask(self.noisy)
        self.target = self.noisy


    def validate_init(self):
        self.noisy = self.data['noisy']
        self.data = self.data['gt']
        self.noisy = self.space_to_depth(self.noisy,block_size=2)


    def validate_process_images(self):
        # unpadding
        denoise_exp = self.depth_to_space(self.exp_denoise,block_size=2)
        # data = self.depth_to_space(self.data,block_size=2)
        data = self.data
        # noisy = self.depth_to_space(self.noisy,block_size=2)
        noisy = self.noisy
        # denoise_exp = self.exp_denoise[:,:,:self.H,:self.W]

        self.denoise01_exp = denoise_exp.permute(0,2,3,1).cpu().data.clamp(0,1).numpy().squeeze(0)
        self.ori01 = data.permute(0,2,3,1).cpu().data.clamp(0,1).numpy().squeeze(0)
        self.noisy01 = self.noisy.permute(0,2,3,1).cpu().data.clamp(0,1).numpy().squeeze(0)

        self.denoise255_exp = tensor2image(denoise_exp)
        self.ori255 = tensor2image(data)
        self.noisy255 = tensor2image(noisy)

    def get_val_result(self):
        # calculate metrics
        denoise_exp_psnr = calculate_psnr(self.ori01.astype(np.float32),self.denoise01_exp.astype(np.float32),1)
        denoise_exp_ssim = calculate_ssim(self.ori01.astype(np.float32)*255.0,self.denoise01_exp.astype(np.float32)*255.0)
        # collect and append result
        curr_validate_result={'img_name':self.data_name,
                                      'denoise_exp_psnr': denoise_exp_psnr,
                                      'denoise_exp_ssim': denoise_exp_ssim,
                                      }
        return curr_validate_result
    def save_results(self,repeat=0,idx=0):
        if (self.is_train and self.validate_opt.save_results) or (self.is_test and self.test_opt.save_results):
            out = 'val_imgs' if self.is_train else 'test_imgs'
            save_dir = os.path.join(self.train_url,out)

            save_path = os.path.join(
                save_dir,
                "{:03d}-{:03d}-{:03d}_exp.png".format(
                    idx,repeat,  self.epoch))
            Image.fromarray(self.denoise255_exp.squeeze()).save(save_path)
            save_path = os.path.join(
                save_dir,
                "{:03d}-{:03d}-{:03d}_clean.png".format(
                    idx,repeat, self.epoch))
            Image.fromarray(self.ori255.squeeze()).save(save_path)  

            save_path = os.path.join(
                save_dir,
                "{:03d}-{:03d}-{:03d}_noisy.png".format(
                    idx,repeat, self.epoch))
            Image.fromarray(self.noisy255.squeeze()).save(save_path) 


class Frequency_B2UB_Model_SupStd(Frequency_B2UB_Model_SIDDRaw):
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

        mask_input = torch.mean(exp_dn,dim=1,keepdim=True)*255.0
        mask = 1-self.generate_upper(mask_input,thres,ws)

        # print(torch.mean(mask))

        # print the ratio
        # print(torch.sum(mask)/torch.sum(torch.ones_like(mask)))

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
        anchor_exp = self.depth_to_space(self.anchor_exp,block_size=2)
        self.anchor01_exp = anchor_exp.permute(0,2,3,1).cpu().data.clamp(0,1).numpy().squeeze(0)
        self.anchor255_exp = tensor2image(anchor_exp)

    def get_val_result(self):
        curr_validate_result = super().get_val_result()
        anchor_exp_psnr = calculate_psnr(self.ori01.astype(np.float32),self.anchor01_exp.astype(np.float32),1)
        anchor_exp_ssim = calculate_ssim(self.ori01.astype(np.float32)*255.0,self.anchor01_exp.astype(np.float32)*255.0)

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
            save_dir = os.path.join(self.train_url,out)

            save_path = os.path.join(
                save_dir,
                "{:03d}-{:03d}-{:03d}_anchor.png".format(
                    idx,repeat,  self.epoch))
            Image.fromarray(self.anchor255_exp.squeeze()).save(save_path)

