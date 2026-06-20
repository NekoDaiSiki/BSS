import os
import time

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

class BlindSpotModel(BaseModel):
    def __init__(self, opt):
        super(BlindSpotModel,self).__init__(opt)
        self.mask = None 

    def create_models(self):
        # masker and noiser are not nn.Module
        self.train_noise_adder = build_from_cfg(self.sub_models_opt.noiser,NOISER)
        self.train_masker = build_from_cfg(self.sub_models_opt.masker, MASKER)
        # NN-submodel
        self.backbone = build_from_cfg(self.sub_models_opt.backbone, BACKBONE)

        # dict of NN-submodels 
        self.sub_models.update({
            'backbone': self.backbone,
        })

    def train_init(self):
        self.time_start = time.time()
        self.data = self.data/255.0 # normlize input
        # pre process data 
        self.noisy = self.train_noise_adder.add_noise(self.data)
        self.input, self.mask = self.train_masker.add_mask(self.noisy)
        self.target = self.noisy

    def train_logging(self):
        if self.curr_iter%self.train_opt.print_freq ==0:
            self.get_msg()
            self.logger.info(self.msg)

    def get_val_result(self,repeat,index):
        pass
    def save_results(self,repeat,index):
        pass
    def validate_cal_metrics(self,repeat,index):
        curr_validate_result = self.get_val_result(repeat,index)
        self.logger.info(curr_validate_result) if self.is_test else print(curr_validate_result)
        self.validate_results.append(curr_validate_result)
        # self.save_results(repeat,index)

    def logger_val_summary(self,avg_results):
        self.logger.info("epoch:{},dn:{:.6f}/{:.6f},exp:{:.6f}/{:.6f},mid:{:.6f}/{:.6f}".format(
                    self.epoch, avg_results['denoise_psnr'], avg_results['denoise_ssim'], 
                        avg_results['denoise_exp_psnr'], avg_results['denoise_exp_ssim'], 
                        avg_results['denoise_mid_psnr'], avg_results['denoise_mid_ssim']))
    def validate_summary(self):
        # logger results
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
            avg_results[k] = avg_results[k]/len_results

        self.logger_val_summary(avg_results)
        
        # reset validate_results to emplty list
        self.validate_results = []



class Noise2Void(BlindSpotModel):
    def train_main_process(self):
        nv,_,_,_ = self.input.shape
        n,c,h,w = self.noisy.shape
        v = int(nv/n)
        self.target = self.target.view(n,1,c,h,w).repeat(1,v,1,1,1).view(-1,c,h,w)
        self.denoise = self.backbone(self.input) #(n*v,c,h,w)
        with torch.no_grad():
            self.exp_denoise = self.backbone(self.noisy)
        
    def train_cal_loss(self):
        diff = (self.denoise - self.target)*self.mask
        loss = torch.mean(diff**2)
        self.loss = loss

    def get_msg(self):
        lr = self.scheduler.get_last_lr()[0]
        time_start = self.time_start
        time_end = time.time()

        dn = self.exp_denoise
        ori = self.data
        distortion = calculate_psnr_from_tensor(ori,dn)
        self.msg = '{:04d} {:05d} lr={:.2e} distortion={:.6f}, Loss_All={:.6f}, Time={:.4f}'.format(
                            self.epoch,self.curr_iter, lr, distortion, self.loss.item(), time_end -time_start)

    def validate_init(self):
        self.data = self.data/255.0
        # add noise
        self.noisy = self.train_noise_adder.add_noise(self.data)
        # padding to square, need to research it
        n,c,h,w = self.data.shape
        self.H,self.W = h,w
        val_sizeH = (h + 31) // 32 * 32
        val_sizeW = (w + 31) // 32 * 32
        self.noisy = F.pad(self.noisy,[0,val_sizeW-w, 0,val_sizeH-h],mode='reflect')

    def validate_main_process(self):
        with torch.no_grad():
            self.denoise = self.backbone(self.noisy)

    def validate_process_images(self):
        # unpadding
        denoise = self.denoise[:,:,:self.H,:self.W]
        self.denoise255 = tensor2image(denoise)
        self.ori255 = tensor2image(self.data)
        self.noisy255 = tensor2image(self.noisy[:,:,:self.H,:self.W])

    def get_val_result(self,repeat,index):
        # calculate metrics
        denoise_psnr = calculate_psnr(self.ori255,self.denoise255)
        denoise_ssim = calculate_ssim(self.ori255,self.denoise255)
        # collect and append result
        curr_validate_result = {'img_name':self.data_name[0],
                                      'denoise_psnr': denoise_psnr,
                                      'denoise_ssim': denoise_ssim,
                                      }
        return curr_validate_result

    def logger_val_summary(self,avg_results):
        self.logger.info("epoch:{},dn:{:.6f}/{:.6f}".format(
                    self.epoch, avg_results['denoise_psnr'], avg_results['denoise_ssim']))
                    
    def save_results(self,repeat=0,idx=0):
        if (self.is_train and self.validate_opt.save_results) or (self.is_test and self.test_opt.save_results):
            out = 'val_imgs' if self.is_train else 'test_imgs'
            opt_testOrVal = self.datasets_opt.validate if self.is_train else self.datasets_opt.test
            name_dataset = opt_testOrVal.data_dir.split('/')[-1]
            save_dir = os.path.join(self.train_url,out,name_dataset)
            mkdir(save_dir)

            save_path = os.path.join(
                save_dir,
                "{:03d}-{:03d}-{:03d}_dn.png".format(
                    idx,repeat, self.epoch))
            Image.fromarray(self.denoise255).convert('RGB').save(save_path)                
            save_path = os.path.join(
                save_dir,
                "{:03d}-{:03d}-{:03d}_clean.png".format(
                    idx,repeat, self.epoch))
            Image.fromarray(self.ori255).convert('RGB').save(save_path)  

            save_path = os.path.join(
                save_dir,
                "{:03d}-{:03d}-{:03d}_noisy.png".format(
                    idx,repeat, self.epoch))
            Image.fromarray(self.noisy255).convert('RGB').save(save_path) 

class Noise2Viod_WeightMask(Noise2Void):
    def train_cal_loss(self):
        diff = (self.denoise - self.target)*self.mask
        loss = torch.mean(diff**2)/torch.mean(self.mask)
        # print(torch.mean(self.mask))

        self.loss = loss

class Noise2Void_SIDDRaw(Noise2Void):
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
        # pre process data 
        self.noisy = self.data
        self.input, self.mask = self.train_masker.add_mask(self.noisy)
        self.target = self.noisy

    def validate_init(self):
        self.noisy = self.data['noisy']
        self.data = self.data['gt']
        self.noisy = self.space_to_depth(self.noisy,block_size=2)
    
    def validate_process_images(self):
        # self.denoise = denoise
        denoise = self.depth_to_space(self.denoise,block_size=2)
        
        self.denoise_01 = denoise.permute(0,2,3,1).cpu().data.clamp(0,1).numpy().squeeze(0)
        self.ori_01 = self.data.permute(0,2,3,1).cpu().data.clamp(0,1).numpy().squeeze(0)

        self.denoise255 = tensor2image(denoise)
        self.ori255 = tensor2image(self.data)
        self.noisy255 = tensor2image(self.noisy)

    def get_val_result(self,index,repeat):
        # calculate metrics
        # denoise_psnr = calculate_psnr(self.ori255,self.denoise255)
        # denoise_ssim = calculate_ssim(self.ori255,self.denoise255)
        denoise_psnr = calculate_psnr(self.ori_01.astype(np.float32),self.denoise_01.astype(np.float32),1)
        denoise_ssim = calculate_ssim(self.ori_01.astype(np.float32)*255,self.denoise_01.astype(np.float32)*255)
        # collect and append result
        curr_validate_result = {'img_name':self.data_name[0],
                                      'denoise_psnr': denoise_psnr,
                                      'denoise_ssim': denoise_ssim,
                                      }
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
                "{:03d}-{:03d}-{:03d}_dn.png".format(
                    idx,repeat, self.epoch))
            Image.fromarray(self.denoise255.squeeze()).save(save_path)                
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

class Noise2Void_WeightLoss(Noise2Void):

    def train_cal_loss(self):
        diff = (self.denoise - self.target)*self.mask
        loss = torch.sum(diff**2)
        num = torch.sum(self.mask)
        self.loss = loss/num

class B2UB_Model(BlindSpotModel):   
    def train_main_process(self):
        n,c,h,w = self.noisy.shape
        denoise = self.backbone(self.input)
        self.denoise = (denoise*self.mask).view(n, -1, c, h, w).sum(dim=1)

        with torch.no_grad():
            self.exp_denoise = self.backbone(self.noisy)

    def get_beta(self):
        Lambda = (self.epoch+1)/self.train_opt.total_epoch
        Thread1 = self.loss_params.thread1
        Thread2 = self.loss_params.thread2
        Lambda1 = self.loss_params.lambda1
        Lambda2 = self.loss_params.lambda2
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

    def train_cal_loss(self):
        diff = self.denoise - self.target
        exp_diff = self.exp_denoise - self.target
        alpha,beta = self.get_beta()
        revisible = diff + beta * exp_diff
        self.beta = beta
        self.loss_reg = alpha * torch.mean(diff**2)
        self.loss_rev = torch.mean(revisible**2)
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

        self.msg = '{:04d} {:05d} lr={:.2e} distortion={:.6f}, distortion_exp={:.6f}, Loss_Reg={:.6f}, Beta={}, Loss_Rev={:.6f}, Loss_All={:.6f}, Time={:.4f}'.format(
                self.epoch,self.curr_iter, lr, distortion, distortion_exp , \
                        self.loss_reg.item(), self.beta, self.loss_rev.item(), self.loss.item(), time_end -time_start)

    def validate_init(self):
        self.data = self.data/255.0
        # add noise
        self.noisy = self.train_noise_adder.add_noise(self.data)
        # padding to square, need to research it
        n,c,h,w = self.data.shape
        self.H,self.W = h,w
        val_size = (max(h, w) + 31) // 32 * 32
        self.noisy = F.pad(self.noisy,[0,val_size-w,0,val_size-h],mode='reflect')
        self.input, self.mask = self.train_masker.add_mask(self.noisy)

    def validate_main_process(self):
        with torch.no_grad():
            n,c,h,w = self.noisy.shape
            denoise = self.backbone(self.input)
            self.denoise = (denoise*self.mask).view(n, -1, c, h, w).sum(dim=1)
            self.exp_denoise = self.backbone(self.noisy)
    
    def validate_process_images(self):
        # unpadding
        alpha,beta = self.get_beta()
        denoise = self.denoise[:,:,:self.H,:self.W]
        denoise_exp = self.exp_denoise[:,:,:self.H,:self.W]
        denoise_mid = (denoise + beta*denoise_exp) / (1 + beta)

        self.denoise255 = tensor2image(denoise)
        self.denoise255_exp = tensor2image(denoise_exp)
        self.denoise255_mid = tensor2image(denoise_mid)
        self.ori255 = tensor2image(self.data)
        self.noisy255 = tensor2image(self.noisy[:,:,:self.H,:self.W])

    def get_val_result(self,repeat,index):
        # calculate metrics
        denoise_psnr = calculate_psnr(self.ori255,self.denoise255)
        denoise_ssim = calculate_ssim(self.ori255,self.denoise255)

        denoise_exp_psnr = calculate_psnr(self.ori255,self.denoise255_exp)
        denoise_exp_ssim = calculate_ssim(self.ori255,self.denoise255_exp)

        denoise_mid_psnr = calculate_psnr(self.ori255,self.denoise255_mid)
        denoise_mid_ssim = calculate_ssim(self.ori255,self.denoise255_mid)
        # collect and append result

        data_name = "{:03d}-{:03d}-{:03d}_{}".format(
                    index,repeat, self.epoch, self.data_name[0])
        curr_validate_result={'img_name':data_name,
                                      'denoise_psnr': denoise_psnr,
                                      'denoise_ssim': denoise_ssim,
                                      'denoise_exp_psnr': denoise_exp_psnr,
                                      'denoise_exp_ssim': denoise_exp_ssim,
                                      'denoise_mid_psnr': denoise_mid_psnr,
                                      'denoise_mid_ssim':denoise_mid_ssim,
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
                "{:03d}-{:03d}-{:03d}_dn.png".format(
                    idx,repeat, self.epoch))
            Image.fromarray(self.denoise255).convert('RGB').save(save_path)
            save_path = os.path.join(
                save_dir,
                "{:03d}-{:03d}-{:03d}_exp.png".format(
                    idx,repeat,  self.epoch))
            Image.fromarray(self.denoise255_exp).convert('RGB').save(save_path)
            save_path = os.path.join(
                save_dir,
                "{:03d}-{:03d}-{:03d}_mid.png".format(
                    idx,repeat, self.epoch))
            Image.fromarray(self.denoise255_mid).convert('RGB').save(save_path)      
                  
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


class B2UB_Model_SIDDRaw(B2UB_Model):
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

    def validate_main_process(self):
        with torch.no_grad():
            self.exp_denoise = self.backbone(self.noisy)


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

    def get_val_result(self,repeat,index):
        # calculate metrics
        denoise_exp_psnr = calculate_psnr(self.ori01.astype(np.float32),self.denoise01_exp.astype(np.float32),1)
        denoise_exp_ssim = calculate_ssim(self.ori01.astype(np.float32)*255.0,self.denoise01_exp.astype(np.float32)*255.0)
        # collect and append result
        curr_validate_result={'img_name':self.data_name[0],
                                      'denoise_exp_psnr': denoise_exp_psnr,
                                      'denoise_exp_ssim': denoise_exp_ssim,
                                      }
        return curr_validate_result
        
    def logger_val_summary(self,avg_results):
        self.logger.info("epoch:{},exp:{:.6f}/{:.6f}".format(
                    self.epoch, 
                        avg_results['denoise_exp_psnr'], avg_results['denoise_exp_ssim'], ))

    def save_results(self,repeat=0,idx=0):
        # print(repeat)
        if (self.is_train and self.validate_opt.save_results) or (self.is_test and self.test_opt.save_results):
            out = 'val_imgs' if self.is_train else 'test_imgs'
            save_dir = os.path.join(self.train_url,out)

            print("{:03d}-{:03d}-{:03d}-{}_exp.png".format(
                    idx, repeat,  self.epoch, self.data_name[0]))
            save_path = os.path.join(
                save_dir,
                "{:03d}-{:03d}-{:03d}--{}_exp.png".format(
                    idx, repeat,  self.epoch,self.data_name[0]))
            Image.fromarray(self.denoise255_exp.squeeze()).save(save_path)
            save_path = os.path.join(
                save_dir,
                "{:03d}-{:03d}-{:03d}--{}_clean.png".format(
                    idx,repeat, self.epoch,self.data_name[0]))
            Image.fromarray(self.ori255.squeeze()).save(save_path)  

            save_path = os.path.join(
                save_dir,
                "{:03d}-{:03d}-{:03d}--{}_noisy.png".format(
                    idx,repeat, self.epoch,self.data_name[0]))
            Image.fromarray(self.noisy255.squeeze()).save(save_path) 



class B2UB_Model_Onlymask(B2UB_Model):
    def train_cal_loss(self):
        super().train_cal_loss()
        self.loss = self.loss_reg
    # def train
    #     super(BlindSpotModel,self).__init__(opt)

class B2UB_MM2(B2UB_Model):
    def train_main_process(self):
        n,c,h,w = self.noisy.shape
        denoise = self.backbone(self.input)
        self.denoise = (denoise*self.mask).view(n, -1, 4,c, h, w).sum(dim=1)
        dn_a1,dn_a2,dn_mu1,dn_mu2 = self.denoise.chunk(4,dim=1)
        dn_a1,dn_a2 = torch.softmax(torch.cat([dn_a1,dn_a2],dim=1),dim=1).chunk(2,dim=1)
        dn_a1,dn_a2,dn_mu1,dn_mu2 = dn_a1.squeeze(1),dn_a2.squeeze(1),dn_mu1.squeeze(1),dn_mu2.squeeze(1)
        self.dn_mixmu = dn_a1*dn_mu1 +dn_a2*dn_mu2

        with torch.no_grad():
            self.exp_denoise = self.backbone(self.noisy)
        exp_dn_a1,exp_dn_a2,exp_dn_mu1,exp_dn_mu2 = self.exp_denoise.view(n, 4,c, h, w).chunk(4,dim=1)
        exp_dn_a1,exp_dn_a2 = torch.softmax(torch.cat([exp_dn_a1,exp_dn_a2],dim=1),dim=1).chunk(2,dim=1)
        exp_dn_a1,exp_dn_a2,exp_dn_mu1,exp_dn_mu2 = exp_dn_a1.squeeze(1),exp_dn_a2.squeeze(1),exp_dn_mu1.squeeze(1),exp_dn_mu2.squeeze(1)
        self.exp_dn_mixmu = dn_a1*exp_dn_mu1 +dn_a2*exp_dn_mu2

        likeli1 = dn_a1*torch.exp(-(self.noisy-dn_mu1)**2/2)
        likeli2 = dn_a2*torch.exp(-(self.noisy-dn_mu2)**2/2)
        self.likelihood = likeli1 + likeli2

        mask = (likeli1>=likeli2).float()
        self.dn_select = mask*dn_mu1 + (1-mask)*dn_mu2
        self.exp_dn_select = mask*exp_dn_mu1 + (1-mask)*exp_dn_mu2

        if self.curr_iter%10==0:
            print(dn_mu1[0,0,64,64].item(),dn_mu2[0,0,64,64].item())
            print(exp_dn_mu1[0,0,64,64].item(),exp_dn_mu2[0,0,64,64].item())
            # print(dn_sigma1[0,0,64,64].item(),dn_sigma2[0,0,64,64].item())
            print(dn_a1[0,0,64,64].item(),dn_a2[0,0,64,64].item())
            print(exp_dn_a1[0,0,64,64].item(),exp_dn_a2[0,0,64,64].item())
            print(self.noisy[0,0,64,64].item())
            print(self.data[0,0,64,64].item())
            print('\n')



    def train_cal_loss(self):
        mixdiff = self.dn_mixmu - self.target
        mixrev = (self.exp_dn_mixmu - self.target).detach()*self.dn_mixmu

        selectdiff = self.dn_mixmu - self.target
        selectrev = (self.exp_dn_select - self.target).detach()*self.dn_select
        alpha,beta = self.get_beta()
        # self.beta = beta
        self.loss_mixdiff = 0 * torch.mean(mixdiff**2)
        self.loss_mixrev = 0 * torch.mean(mixrev)

        self.loss_selectdiff =0 * torch.mean(selectdiff**2)
        self.selectrev = 2 *  torch.mean(selectrev)

        self.loss_likeli = -1*torch.mean(torch.log(self.likelihood.clamp(min=1e-9)))
        self.loss = self.loss_mixdiff + self.loss_mixrev + self.loss_mixdiff + self.selectrev +self.loss_likeli    

    def get_msg(self):
        lr = self.scheduler.get_last_lr()[0]
        time_start = self.time_start
        time_end = time.time()
        ori = self.data
        dn_mix = calculate_psnr_from_tensor(ori,self.dn_mixmu)
        exp_dn_mix = calculate_psnr_from_tensor(ori,self.exp_dn_mixmu)
        dn_select = calculate_psnr_from_tensor(ori,self.dn_select)
        exp_dn_select = calculate_psnr_from_tensor(ori,self.exp_dn_select)


        self.msg = '{:04d} {:05d} lr={:.2e} dn_mix={:.6f}, exp_dn_mix={:.6f}, dn_select={:.6f}, exp_dn_select={:.6f},Loss_mixdiff={:.6f},Loss_likeli={:.6f} Loss_All={:.6f}, Time={:.4f}'.format(
                self.epoch,self.curr_iter, lr, dn_mix, exp_dn_mix ,dn_select,exp_dn_select, \
                     self.loss_mixdiff.item(),self.loss_likeli.item() ,self.loss.item(), time_end -time_start)

    def validate_main_process(self):
        with torch.no_grad():
            self.train_main_process()

    def validate_process_images(self):
        # unpadding
        denoise_mix = self.exp_dn_mixmu[:,:,:self.H,:self.W]
        denoise_selecct = self.exp_dn_select[:,:,:self.H,:self.W]

        self.denoise255_mix = tensor2image(denoise_mix)
        self.denoise255_selecct = tensor2image(denoise_selecct)
        self.ori255 = tensor2image(self.data)
        self.noisy255 = tensor2image(self.noisy[:,:,:self.H,:self.W])

    def get_val_result(self):
        # calculate metrics
        denoise_mix_psnr = calculate_psnr(self.ori255,self.denoise255_mix)
        denoise_mix_ssim = calculate_ssim(self.ori255,self.denoise255_mix)

        denoise_selecct_psnr = calculate_psnr(self.ori255,self.denoise255_selecct)
        denoise_selecct_ssim = calculate_ssim(self.ori255,self.denoise255_selecct)
        # collect and append result
        curr_validate_result={'img_name':self.data_name,
                                      'denoise_mix_psnr': denoise_mix_psnr,
                                      'denoise_mix_ssim': denoise_mix_ssim,
                                      'denoise_selecct_psnr': denoise_selecct_psnr,
                                      'denoise_selecct_ssim': denoise_selecct_ssim,
                                      }
        return curr_validate_result

    def logger_val_summary(self,avg_results):
        self.logger.info("epoch:{},mix:{:.6f}/{:.6f},select:{:.6f}/{:.6f}".format(
                    self.epoch, avg_results['denoise_mix_psnr'], avg_results['denoise_mix_ssim'], 
                        avg_results['denoise_selecct_psnr'], avg_results['denoise_selecct_ssim'], ))


class B2UB_MM2_new(B2UB_Model):

    # def onehot_from_logits(self, logits, eps=0.0):
    #     """
    #     Given batch of logits, return one-hot sample using epsilon greedy strategy
    #     (based on given epsilon)
    #     """
    #     # get best (according to current policy) actions in one-hot form
    #     argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()

    #     # 探索率为0，则直接以概率大小选择最优操作
    #     if eps == 0.0:
    #         return argmax_acs

    #     # get random actions in one-hot form
    #     rand_acs = torch.eye(logits.shape[1])[[np.random.choice(
    #         range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)

    #     # 探索率不为0，则chooses between best and random actions using epsilon greedy
    #     return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
    #                         enumerate(torch.rand(logits.shape[0]))])

    def train_main_process(self):
        n,c,h,w = self.noisy.shape
        denoise = self.backbone(self.input)
        self.denoise = (denoise*self.mask).view(n, -1, 6,c, h, w).sum(dim=1)
        dn_a1,dn_a2,dn_mu1,dn_mu2,dn_sigma1,dn_sigma2 = self.denoise.chunk(6,dim=1)
        dn_a1,dn_a2 = torch.softmax(torch.cat([dn_a1,dn_a2],dim=1),dim=1).chunk(2,dim=1)
        dn_a1,dn_a2,dn_mu1,dn_mu2,dn_sigma1,dn_sigma2 = dn_a1.squeeze(1),dn_a2.squeeze(1),dn_mu1.squeeze(1),dn_mu2.squeeze(1),dn_sigma1.squeeze(1),dn_sigma2.squeeze(1)
        self.dn_mixmu = dn_a1*dn_mu1 +dn_a2*dn_mu2

        with torch.no_grad():
            self.exp_denoise = self.backbone(self.noisy)
        exp_dn_a1,exp_dn_a2,exp_dn_mu1,exp_dn_mu2,_,_ = self.exp_denoise.chunk(6,dim=1)
        exp_dn_a1,exp_dn_a2 = torch.softmax(torch.cat([exp_dn_a1,exp_dn_a2],dim=1),dim=1).chunk(2,dim=1)
        exp_dn_a1,exp_dn_a2,exp_dn_mu1,exp_dn_mu2 = exp_dn_a1.squeeze(1),exp_dn_a2.squeeze(1),exp_dn_mu1.squeeze(1),exp_dn_mu2.squeeze(1)
        self.exp_dn_mixmu = dn_a1*exp_dn_mu1 +dn_a2*exp_dn_mu2

        dn_sigma1,dn_sigma2 = dn_sigma1.abs().clamp(min=0.01),dn_sigma2.abs().clamp(min=0.01)
        likeli1 = dn_a1*(1/dn_sigma1)*torch.exp(-(self.noisy-dn_mu1)**2/2/dn_sigma1**2)
        likeli2 = dn_a2*(1/dn_sigma2)*torch.exp(-(self.noisy-dn_mu2)**2/2/dn_sigma2**2)
        self.likelihood = likeli1 + likeli2

        if self.main_model_opt.select_mode == 'm0':
            mask = (likeli1>=likeli2).float()
            self.dn_select = mask*dn_mu1 + (1-mask)*dn_mu2
            self.exp_dn_select = mask*exp_dn_mu1 + (1-mask)*exp_dn_mu2
        elif self.main_model_opt.select_mode == 'm0_1':
            cat_like = torch.cat([likeli1.unsqueeze(4),likeli2.unsqueeze(4)],dim=4).view(-1,2)
            cat_like = torch.softmax(cat_like,dim=1)
            idx = torch.multinomial(cat_like, 1, replacement=False)
            # print(idx.shape)
            n,c,h,w = self.noisy.shape
            idx = idx.view(n,c,h,w,1)
            self.dn_select = torch.gather(torch.cat([dn_mu1.unsqueeze(4),dn_mu1.unsqueeze(4)],dim=4),dim=4,index=idx).squeeze(4)
            self.exp_dn_select = torch.gather(torch.cat([exp_dn_mu1.unsqueeze(4),exp_dn_mu2.unsqueeze(4)],dim=4),dim=4,index=idx).squeeze(4)
        elif self.main_model_opt.select_mode == 'm1':
            mask = F.gumbel_softmax(torch.cat([likeli1.unsqueeze(1),likeli2.unsqueeze(1)],dim=1),hard=True,dim=1)[:,0,:,:,:]
            self.dn_select = mask*dn_mu1 + (1-mask)*dn_mu2
            self.exp_dn_select = mask*exp_dn_mu1 + (1-mask)*exp_dn_mu2
        elif self.main_model_opt.select_mode == 'm2':
            mask = ((self.noisy-dn_mu1).abs() < (self.noisy - dn_mu2).abs()).float()
            self.dn_select = mask*dn_mu1 + (1-mask)*dn_mu2
            self.exp_dn_select = mask*exp_dn_mu1 + (1-mask)*exp_dn_mu2
        elif self.main_model_opt.select_mode == 'm3':
            w1 = (likeli1+1e-5)/(likeli1+likeli2+2e-5)
            w2 = 1-w1
            self.dn_select = w1*dn_mu1 + w2*dn_mu2
            self.exp_dn_select = w1*exp_dn_mu1 + w2*exp_dn_mu2
        elif self.main_model_opt.select_mode == 'm3_1':
            w1 = (likeli1+1e-13)/(likeli1+likeli2+2e-13).detach() # 截断梯度
            w2 = 1-w1
            self.dn_select = w1*dn_mu1 + w2*dn_mu2
            self.exp_dn_select = w1*exp_dn_mu1 + w2*exp_dn_mu2
        
        elif self.main_model_opt.select_mode == 'm4':
            w1 = (likeli1+1e-13)/(likeli1+likeli2+2e-13).detach()
            w2 = 1-w1
            w = torch.cat([w1.unsqueeze(4),w2.unsqueeze(4)],dim=4).view(-1,2)
            idx = torch.multinomial(w, 1, replacement=False)
            # print(idx.shape)
            n,c,h,w = self.noisy.shape
            idx = idx.view(n,c,h,w,1)
            self.dn_select = torch.gather(torch.cat([dn_mu1.unsqueeze(4),dn_mu1.unsqueeze(4)],dim=4),dim=4,index=idx).squeeze(4)
            self.exp_dn_select = torch.gather(torch.cat([exp_dn_mu1.unsqueeze(4),exp_dn_mu2.unsqueeze(4)],dim=4),dim=4,index=idx).squeeze(4)


        # gt_select_mask = ((self.data-dn_mu1).abs() < (self.data - dn_mu2).abs()).float()
        # self.dn_select_gt = gt_select_mask*dn_mu1 + (1-gt_select_mask)*dn_mu2
        # self.exp_dn_select_gt = gt_select_mask*exp_dn_mu1 + (1-gt_select_mask)*exp_dn_mu2

        self.distance = (dn_mu1 - dn_mu2).abs()
        dis_mask = (self.distance < 0.2).float()
        self.distance = self.distance * dis_mask
        # if self.curr_iter%10==0:
            # print(dn_mu1[0,0,64,64].item(),dn_mu2[0,0,64,64].item())
            # print(dn_sigma1[0,0,64,64].item(),dn_sigma2[0,0,64,64].item())
            # print(dn_a1[0,0,64,64].item(),dn_a2[0,0,64,64].item())
            # print(exp_dn_mu1[0,0,64,64].item(),exp_dn_mu2[0,0,64,64].item())
            # print(self.noisy[0,0,64,64].item())
            # print(self.data[0,0,64,64].item())
            # print('\n')


    def get_coefficient(self):
        loss_params = self.train_opt.loss_params
        a1 = loss_params.a1
        a2 = loss_params.a2
        b1 = loss_params.b1
        b2 = loss_params.b2
        c1 = loss_params.c1
        return a1,a2,b1,b2,c1


    def train_cal_loss(self):
        mixdiff = self.dn_mixmu - self.target
        mixrev = (self.exp_dn_mixmu - self.target).detach()*self.dn_mixmu

        selectdiff = self.dn_mixmu - self.target
        selectrev = (self.exp_dn_select - self.target).detach()*self.dn_select
        # alpha,beta = self.get_beta()
        a1,a2,b1,b2,c1 = self.get_coefficient()
        # self.beta = beta
        self.loss_mixdiff = a1 * torch.mean(mixdiff**2)
        self.loss_mixrev = a2 * torch.mean(mixrev)
        self.loss_selectdiff =b1 * torch.mean(selectdiff**2)
        self.selectrev = b2 *  torch.mean(selectrev)
        self.distance_loss = -0*torch.mean((self.distance))
        self.loss_likeli = -1*c1*torch.mean(torch.log(self.likelihood.clamp(min=1e-9)))
        self.loss = self.loss_mixdiff + self.loss_mixrev + self.loss_mixdiff + self.selectrev +self.loss_likeli  + self.distance_loss

    def get_msg(self):
        lr = self.scheduler.get_last_lr()[0]
        time_start = self.time_start
        time_end = time.time()
        ori = self.data
        dn_mix = calculate_psnr_from_tensor(ori,self.dn_mixmu)
        exp_dn_mix = calculate_psnr_from_tensor(ori,self.exp_dn_mixmu)
        dn_select = calculate_psnr_from_tensor(ori,self.dn_select)
        exp_dn_select = calculate_psnr_from_tensor(ori,self.exp_dn_select)
        # dn_select_gt = calculate_psnr_from_tensor(ori,self.dn_select_gt)
        # exp_dn_select_gt = calculate_psnr_from_tensor(ori,self.exp_dn_select_gt)


        self.msg = '{:04d} {:05d} lr={:.2e} dn_mix={:.6f}, exp_dn_mix={:.6f}, dn_select={:.6f}, exp_dn_select={:.6f},  Loss_SecRev={:.6f},  Loss_Like={:.6f}, Loss_All={:.6f}, Time={:.4f}'.format(
                self.epoch,self.curr_iter, lr, dn_mix, exp_dn_mix ,dn_select,exp_dn_select, \
                    self.selectrev.item(),self.loss_likeli.item(), self.loss.item(), time_end -time_start)

    def validate_init(self):
        self.data = self.data/255.0
        # add noise
        self.noisy = self.train_noise_adder.add_noise(self.data)
        # padding to square, need to research it
        n,c,h,w = self.data.shape
        self.H,self.W = h,w
        val_size = (max(h, w) + 31) // 32 * 32
        self.noisy = F.pad(self.noisy,[0,val_size-w,0,val_size-h],mode='reflect')
        self.data = F.pad(self.data,[0,val_size-w,0,val_size-h],mode='reflect')
        self.input, self.mask = self.train_masker.add_mask(self.noisy)

    def validate_main_process(self):
        with torch.no_grad():
            self.train_main_process()

    def validate_process_images(self):
        # unpadding
        denoise_mix = self.exp_dn_mixmu[:,:,:self.H,:self.W]
        denoise_selecct = self.exp_dn_select[:,:,:self.H,:self.W]

        self.denoise255_mix = tensor2image(denoise_mix)
        self.denoise255_selecct = tensor2image(denoise_selecct)
        self.ori255 = tensor2image(self.data[:,:,:self.H,:self.W])
        self.noisy255 = tensor2image(self.noisy[:,:,:self.H,:self.W])

    def get_val_result(self):
        # calculate metrics
        denoise_mix_psnr = calculate_psnr(self.ori255,self.denoise255_mix)
        denoise_mix_ssim = calculate_ssim(self.ori255,self.denoise255_mix)

        denoise_selecct_psnr = calculate_psnr(self.ori255,self.denoise255_selecct)
        denoise_selecct_ssim = calculate_ssim(self.ori255,self.denoise255_selecct)
        # collect and append result
        curr_validate_result={'img_name':self.data_name,
                                      'denoise_mix_psnr': denoise_mix_psnr,
                                      'denoise_mix_ssim': denoise_mix_ssim,
                                      'denoise_selecct_psnr': denoise_selecct_psnr,
                                      'denoise_selecct_ssim': denoise_selecct_ssim,
                                      }
        return curr_validate_result

    def logger_val_summary(self,avg_results):
        self.logger.info("epoch:{},mix:{:.6f}/{:.6f},select:{:.6f}/{:.6f}".format(
                    self.epoch, avg_results['denoise_mix_psnr'], avg_results['denoise_mix_ssim'], 
                        avg_results['denoise_selecct_psnr'], avg_results['denoise_selecct_ssim'], ))

class B2UB_MM(B2UB_Model):
    def train_main_process(self):
        n,c,h,w = self.noisy.shape
        denoise = self.backbone(self.input)
        self.denoise = (denoise*self.mask).view(n, -1, 3,c, h, w).sum(dim=1)
        
        diff = self.denoise - self.noisy.unsqueeze(1) #(n,3,c,h,w)
        mask = F.gumbel_softmax(- diff.abs()*3,hard=True,dim=1).detach()
        self.denoise = (mask* self.denoise).sum(1)

        with torch.no_grad():
            self.exp_denoise = self.backbone(self.noisy).view(n, 3,c, h, w)    
            self.exp_denoise = (self.exp_denoise*mask).sum(1)

    def validate_main_process(self):
        with torch.no_grad():
            self.train_main_process()
 
class B2UB_Model_Padding(B2UB_Model):
    def train_init(self):
        self.time_start = time.time()
        self.data = self.data/255.0 # normlize input
        # pre process data 
        self.noisy = self.train_noise_adder.add_noise(self.data)
        # self.input, self.mask = self.train_masker.add_mask(self.noisy)
        self.target = self.noisy

    def train_main_process(self):
        n,c,h,w = self.noisy.shape
        self.exp_denoise = self.backbone(self.noisy)
        pads = []
        kernel = [
            torch.tensor([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], (0.5, 1.0, 0.5)],device=self.device).unsqueeze(0).unsqueeze(0),
            torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], (0.0, 1.0, 0.0)],device=self.device).unsqueeze(0).unsqueeze(0),
            torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0], (1.0, 0.0, 1.0)],device=self.device).unsqueeze(0).unsqueeze(0),
        ]
        for k in kernel:
            k = k / k.sum()
            padding = F.conv2d(
                self.noisy.view(n*c,1,h,w), k, stride=1, padding=3//2).unsqueeze(0)
            pads.append(padding)
        pads = torch.cat(pads,dim=0)
        diff = (self.data.view(n*c,1,h,w).unsqueeze(0).detach() - pads).abs()
        mask = F.gumbel_softmax(-diff*100,tau=1,hard=True,dim=0)
        padding = (mask * pads).sum(0).view(n,c,h,w)
        # padding = padding + 20/255*torch.randn_like(padding)
        # 一个新的开关
        # padding = pads[0,...].view(n,c,h,w)

        self.input, self.mask = self.train_masker.add_mask(self.noisy,padding)
        denoise = self.backbone(self.input)
        self.denoise = (denoise*self.mask).view(n, -1, c, h, w).sum(dim=1)
    def validate_init(self):
        self.data = self.data/255.0
        # add noise
        self.noisy = self.train_noise_adder.add_noise(self.data)
        # padding to square, need to research it
        n,c,h,w = self.data.shape
        self.H,self.W = h,w
        val_size = (max(h, w) + 31) // 32 * 32
        self.noisy = F.pad(self.noisy,[0,val_size-w,0,val_size-h],mode='reflect')
        
    def train_cal_loss(self):
        diff = self.denoise - self.target
        noise = self.target -self.exp_denoise
        exp_diff = (self.exp_denoise - self.target).detach()*self.denoise
        alpha,beta = self.get_beta()
        revisible = exp_diff
        self.beta = beta
        self.loss_reg = alpha * torch.mean(diff**2)
        self.loss_rev = 0*torch.mean(revisible)
        self.loss_noise = 0*torch.mean(-noise**2)
        self.loss = self.loss_reg + self.loss_rev +self.loss_noise
    
    def validate_main_process(self):
        with torch.no_grad():
            self.train_main_process()

class B2UB_Model_LearnMask(B2UB_Model_Padding):
    # 一个可学习的mask来消除盲点和非盲点之间的gap
    def create_models(self):
        super().create_models()
        self.masknet = build_from_cfg(self.sub_models_opt.masknet,HEAD)
        self.sub_models.update({
            'masknet': self.masknet,
        })

    def prior_padding(self,tensor):
        n, c, h, w = tensor.shape
        kernel = torch.tensor([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], (0.5, 1.0, 0.5)],device=self.device).unsqueeze(0).unsqueeze(0)
        kernel = kernel / kernel.sum()
        padding = F.conv2d(
                    tensor.view(n*c, 1, h, w), kernel, stride=1, padding=3//2)
        padding = padding.view_as(tensor)
        return padding

    def train_main_process(self):
        n,c,h,w = self.noisy.shape
        with torch.no_grad():
            self.exp_denoise = self.backbone(self.noisy).detach()
        # 3x3 prior kernel embeeding
        self.prpadding = self.prior_padding(self.noisy)
        if self.train_opt.loss_mode==0:
            lean_padding = self.masknet(self.noisy)
        elif self.train_opt.loss_mode ==1:
            lean_padding = (self.exp_denoise.clamp(0,1)).round()
        elif self.train_opt.loss_mode ==2:
            lean_padding = (self.exp_denoise.clamp(0,1)*10).round()/10 + 0.1*(2*torch.rand_like(self.exp_denoise)-1)
        elif self.train_opt.loss_mode ==3:
            lean_padding = ((self.exp_denoise-self.prpadding)*5).round()/5 + self.prpadding

        elif self.train_opt.loss_mode ==4:
            exp_noisy = self.target - self.exp_denoise
            exp_noisy1,exp_noisy2 = torch.chunk(exp_noisy,2)
            exp_noisy_r = torch.cat([exp_noisy2,exp_noisy1],dim=0)
            lean_padding = self.exp_denoise + exp_noisy_r
        elif self.train_opt.loss_mode ==5:
            lean_padding = self.data + 25/255.0*torch.randn_like(self.data)

        elif self.train_opt.loss_mode ==6:
            kernel = torch.tensor([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], (0.5, 1.0, 0.5)],device=self.device).unsqueeze(0).unsqueeze(0).repeat(3,3,1,1)
            kernel = kernel / kernel.sum()*3
            lean_padding = F.conv2d(
                    self.noisy, kernel, stride=1, padding=3//2)
        mix1 = self.train_opt.mix1
        mix2 = self.train_opt.mix2
        self.padding = mix1*self.prpadding + mix2*lean_padding
        self.input, self.mask = self.train_masker.add_mask(self.noisy,self.padding)
        denoise = self.backbone(self.input)
        self.denoise = (denoise*self.mask).view(n, -1, c, h, w).sum(dim=1)

    def train_cal_loss(self):
        diff = self.denoise - self.target
        exp_diff = self.exp_denoise - self.target
        alpha,beta = self.get_beta()
        revisible = diff + beta * exp_diff
        self.beta = beta
        gamma = self.loss_params.gamma
        self.loss_reg = alpha * torch.mean(diff**2)
        self.loss_rev = torch.mean(revisible**2)
        self.mask_loss = gamma*torch.mean((self.target - self.padding)**2)
        self.loss = self.loss_reg + self.loss_rev +self.mask_loss

    def validate_main_process(self):
        with torch.no_grad():
            n,c,h,w = self.noisy.shape
            self.exp_denoise = self.backbone(self.noisy)
            prior_padding = self.prior_padding(self.noisy)
            lean_padding = self.masknet(self.noisy)
            mixcoff = self.train_opt.mix_coff
            self.padding = prior_padding + (1-mixcoff)*lean_padding
            self.input, self.mask = self.train_masker.add_mask(self.noisy,self.padding)
            denoise = self.backbone(self.input)
            self.denoise = (denoise*self.mask).view(n, -1, c, h, w).sum(dim=1)

    def get_msg(self):
        lr = self.scheduler.get_last_lr()[0]

        time_start = self.time_start
        time_end = time.time()

        dn = self.denoise
        dn_exp = self.exp_denoise
        ori = self.data
        distortion = calculate_psnr_from_tensor(ori,dn)
        distortion_exp = calculate_psnr_from_tensor(ori,dn_exp)

        self.msg = '{:04d} {:05d} lr={:.2e} distortion={:.6f}, distortion_exp={:.6f}, Loss_Reg={:.6f}, Beta={}, Loss_Rev={:.6f}, Loss_mask={:.6f}, Loss_All={:.6f}, Time={:.4f}'.format(
                self.epoch,self.curr_iter, lr, distortion, distortion_exp , \
                        self.loss_reg.item(), self.beta, self.loss_rev.item(),self.mask_loss.item(), self.loss.item(), time_end -time_start)


class B2UB_Momentum(B2UB_Model):
    def __init__(self, opt):
        super().__init__(opt)
        self.m_list = self.train_opt.m_list #0.999
        self.stage_list = self.train_opt.stage_list
    def train_epoch_init(self, train_logger, epoch):
        super().train_epoch_init(train_logger, epoch)
        for i in range(len(self.stage_list)):
            if self.epoch>=self.stage_list[i]:
                self.curr_stage = i

    def create_models(self):
        super().create_models()
        self.backbone_m = build_from_cfg(self.sub_models_opt.backbone_m, BACKBONE)

        # dict of NN-submodels 
        self.sub_models.update({
            'backbone_m': self.backbone_m,
        })   

    def init_models(self):
        # init backbone_m with backbone
        for param, param_m in zip(self.backbone.parameters(), self.backbone_m.parameters()):
            param_m.data = param.data.clone()

    @torch.no_grad()
    def momentum_update_key_encoder(self,model,model_m):
        m = self.m_list[self.curr_stage]
        for param, param_m in zip(model.parameters(), model_m.parameters()):
            param_m.data = param_m.data * m + param.data * (1. - m)
        self.m = m

    def optimize_params(self):
        super().optimize_params()
        self.momentum_update_key_encoder(self.backbone,self.backbone_m)

    def train_main_process(self):
        n,c,h,w = self.noisy.shape
        denoise = self.backbone(self.input)
        self.denoise = (denoise*self.mask).view(n, -1, c, h, w).sum(dim=1)
        with torch.no_grad():
            self.exp_denoise = self.backbone_m(self.noisy)
            self.exp_denoise_ = self.backbone(self.noisy)
            # self.exp_denoise = 0.5*(self.exp_denoise+self.exp_denoise_)
            self.exp_denoise = self.exp_denoise_
    def validate_main_process(self):
        with torch.no_grad():
            n,c,h,w = self.noisy.shape
            denoise = self.backbone(self.input)
            self.denoise = (denoise*self.mask).view(n, -1, c, h, w).sum(dim=1)
            self.exp_denoise = self.backbone_m(self.noisy)
            self.exp_denoise_ = self.backbone(self.noisy)

    def get_msg(self):
        lr = self.scheduler.get_last_lr()[0]

        time_start = self.time_start
        time_end = time.time()

        dn = self.denoise
        dn_exp = self.exp_denoise
        ori = self.data
        distortion = calculate_psnr_from_tensor(ori,dn)
        distortion_exp = calculate_psnr_from_tensor(ori,dn_exp)
        distortion_exp_ = calculate_psnr_from_tensor(ori,self.exp_denoise_)

        self.msg = '{:04d} {:05d} lr={:.2e} distortion={:.6f}, distortion_exp={:.6f}, distortion_exp_={:6f}, Loss_Reg={:.6f}, Beta={}, m={}, Loss_Rev={:.6f}, Loss_All={:.6f}, Time={:.4f}'.format(
                self.epoch,self.curr_iter, lr, distortion, distortion_exp,distortion_exp_ , \
                        self.loss_reg.item(), self.beta, self.m, self.loss_rev.item(), self.loss.item(), time_end -time_start)



class B2UB_Loss_Explore(B2UB_Model):
    # 在训好的模型上finetune
    def train_main_process(self):
        n,c,h,w = self.noisy.shape
        denoise = self.backbone(self.input)
        self.denoise = (denoise*self.mask).view(n, -1, c, h, w).sum(dim=1)
        self.exp_denoise = self.backbone(self.noisy)
        
        _,beta = self.get_beta()
        self.mix_denoise = (self.denoise + beta*self.exp_denoise)/(1+beta)

    def get_beta(self):
        alpha = self.loss_params.alpha
        beta = self.loss_params.beta
        self.beta = beta
        return alpha,beta 
    
    def train_cal_loss(self):
        alpha,beta = self.get_beta()
        gamma = self.loss_params.gamma if self.loss_params.gamma else 0
        if self.loss_params.losstype == 'mode0':
            # alpha=2 beta = 40 is the ori loss function
            alpha,beta = 2,40
            self.loss_reg = alpha* torch.mean((self.denoise - self.target)**2)
            self.loss_rev = beta* torch.mean(self.denoise*(self.exp_denoise.detach()-self.target))
            self.loss = self.loss_reg + self.loss_rev
        elif self.loss_params.losstype == 'mode1':
            self.loss_reg = alpha* torch.mean((self.denoise - self.target)**2)
            self.loss_rev = beta* torch.mean(self.denoise*(self.exp_denoise.detach()-self.target))
            self.loss_dic = gamma*torch.mean(self.exp_denoise*(self.mix_denoise.detach()-self.target))
            self.loss = self.loss_reg + self.loss_rev + self.loss_dic

        elif self.loss_params.losstype == 'mode1_1':
            self.loss_reg = alpha* torch.mean((self.denoise - self.target)**2)
            self.loss_rev = beta* torch.mean(self.denoise*(self.exp_denoise.detach()-self.target))
            self.loss_dic = gamma*torch.mean((self.exp_denoise-self.denoise)).abs()
            self.loss = self.loss_reg + self.loss_rev + self.loss_dic

        elif self.loss_params.losstype == 'mode1_2':
            self.loss_reg = alpha* torch.mean((self.denoise - self.target)**2)
            self.loss_rev = beta* torch.mean(self.denoise*(self.exp_denoise.detach()-self.target))
            # atten_mask = F.sigmoid((self.exp_denoise - self.denoise).abs()).detach()
            self.loss_dic = gamma*torch.mean((self.exp_denoise*(self.target - self.exp_denoise).detach())).abs()
            # self.loss_dic = -torch.mean(atten_mask*(self.target-self.exp_denoise)**2)
            self.loss = self.loss_reg + self.loss_rev + self.loss_dic

        elif self.loss_params.losstype == 'mode2':
            alpha,beta = 0,1
            self.loss_reg = alpha* torch.mean((self.denoise - self.target)**2)
            self.loss_rev = beta* torch.mean(self.exp_denoise*(self.exp_denoise.detach()-self.target))
            self.loss = self.loss_reg + self.loss_rev

    def get_msg(self):
        lr = self.scheduler.get_last_lr()[0]

        time_start = self.time_start
        time_end = time.time()

        dn = self.denoise
        dn_exp = self.exp_denoise
        ori = self.data
        mix = self.mix_denoise
        distortion = calculate_psnr_from_tensor(ori,dn)
        distortion_exp = calculate_psnr_from_tensor(ori,dn_exp)
        distortion_mix = calculate_psnr_from_tensor(ori,mix)
        self.msg = '{:04d} {:05d} lr={:.2e} distortion={:.6f}, distortion_exp={:.6f}, distortion_mix={:.6f},  Loss_Reg={:.6f}, Beta={}, Loss_Rev={:.6f}, Loss_All={:.6f}, Time={:.4f}'.format(
                self.epoch,self.curr_iter, lr, distortion, distortion_exp, distortion_mix , \
                        self.loss_reg.item(), self.beta, self.loss_rev.item(), self.loss.item(), time_end -time_start)
        if self.loss_params.losstype == 'mode1_2':
            self.msg = '{:04d} {:05d} lr={:.2e} distortion={:.6f}, distortion_exp={:.6f}, distortion_mix={:.6f},  Loss_Reg={:.6f}, Beta={}, Loss_Rev={:.6f}, Loss_Dic={:.6f}, Loss_All={:.6f}, Time={:.4f}'.format(
                    self.epoch,self.curr_iter, lr, distortion, distortion_exp, distortion_mix , \
                            self.loss_reg.item(), self.beta, self.loss_rev.item(), self.loss_dic , self.loss.item(), time_end -time_start)

class B2UB_Enhance(B2UB_Model):

    def train_main_process(self):
        n,c,h,w = self.noisy.shape
        denoise = self.backbone(self.input)
        self.denoise = (denoise*self.mask).view(n, -1, c, h, w).sum(dim=1)
        self.exp_denoise = self.backbone(self.noisy)


    def get_beta(self):
        alpha = self.loss_params.alpha
        beta = self.loss_params.beta
        gamma = self.loss_params.gamma

        return alpha, beta, gamma

    def train_cal_loss(self):
        diff = self.denoise - self.target
        exp_diff = self.exp_denoise.detach() - self.target
        alpha,beta,gamma = self.get_beta()
        revisible = diff + beta * exp_diff
        self.beta = beta
        self.loss_reg = alpha * torch.mean(diff**2)
        self.loss_rev = torch.mean(revisible**2)
        # self.loss_dir = gamma*torch.mean(self.exp_denoise*(self.exp_denoise.detach()-self.target))
        self.loss_direct = gamma*(torch.mean(self.exp_denoise*(self.exp_denoise.detach()-self.target) +self.exp_denoise.detach()*(self.exp_denoise-self.target)))
        self.loss = self.loss_reg + self.loss_rev + self.loss_direct

    def train_logging(self):
        if self.curr_iter%self.train_opt.print_freq ==0:
            lr = self.scheduler.get_last_lr()[0]

            time_start = self.time_start
            time_end = time.time()

            dn = self.denoise
            dn_exp = self.exp_denoise
            ori = self.data
            distortion = calculate_psnr_from_tensor(ori,dn)
            distortion_exp = calculate_psnr_from_tensor(ori,dn_exp)

            self.msg = '{:04d} {:05d} lr={:.2e} distortion={:.6f}, distortion_exp={:.6f}, Loss_Reg={:.6f}, Beta={}, Loss_Rev={:.6f}, Loss_Dic={:.6f},Loss_All={:.6f}, Time={:.4f}'.format(
                    self.epoch,self.curr_iter, lr, distortion, distortion_exp , \
                            self.loss_reg.item(), self.beta, self.loss_rev.item(),self.loss_direct.item(), self.loss.item(), time_end -time_start)
            self.logger.info(self.msg)

    def validate_process_images(self):
        # unpadding
        alpha,beta,gamma = self.get_beta()
        denoise = self.denoise[:,:,:self.H,:self.W]
        denoise_exp = self.exp_denoise[:,:,:self.H,:self.W]
        denoise_mid = (denoise + beta*denoise_exp) / (1 + beta)

        self.denoise255 = tensor2image(denoise)
        self.denoise255_exp = tensor2image(denoise_exp)
        self.denoise255_mid = tensor2image(denoise_mid)
        self.ori255 = tensor2image(self.data)
        self.noisy255 = tensor2image(self.noisy[:,:,:self.H,:self.W])

class B2UB_Enhance_Head(B2UB_Enhance):
    def create_models(self):
        super().create_models()
        self.head = build_from_cfg(self.sub_models_opt.head,HEAD)
        self.sub_models.update({
            'head': self.head,
        })
    
    def train_main_process(self):
        n,c,h,w = self.noisy.shape
        denoise = self.backbone(self.input)
        self.denoise = (denoise*self.mask).view(n, -1, c, h, w).sum(dim=1)
        self.denoise_bone =  self.backbone(self.noisy).detach()
        self.exp_denoise = self.head(self.denoise_bone)

    def train_cal_loss(self):
        diff = self.denoise - self.target
        exp_diff = self.exp_denoise - self.target
        alpha,beta,gamma = self.get_beta()
        revisible = diff + beta * exp_diff
        self.beta = beta
        self.loss_reg = alpha * torch.mean(diff**2)
        self.loss_rev = torch.mean(revisible**2)
        # self.loss_dir = gamma*torch.mean(self.exp_denoise*(self.exp_denoise.detach()-self.target))
        self.loss_direct = gamma*(torch.mean((self.exp_denoise-self.denoise_bone)**2))
        self.loss = self.loss_reg + self.loss_rev + self.loss_direct

    def train_logging(self):
        if self.curr_iter%self.train_opt.print_freq ==0:
            lr = self.scheduler.get_last_lr()[0]

            time_start = self.time_start
            time_end = time.time()

            dn = self.denoise
            dn_bone = self.denoise_bone
            dn_exp = self.exp_denoise
            ori = self.data
            distortion = calculate_psnr_from_tensor(ori,dn)
            distortion_bone = calculate_psnr_from_tensor(ori,dn_bone)
            distortion_exp = calculate_psnr_from_tensor(ori,dn_exp)

            self.msg = '{:04d} {:05d} lr={:.2e} distortion={:.6f},distortion_bone={:.6f} distortion_exp={:.6f}, Loss_Reg={:.6f}, Beta={}, Loss_Rev={:.6f},Loss_Dic={:.6f}, Loss_All={:.6f}, Time={:.4f}'.format(
                    self.epoch,self.curr_iter, lr, distortion, distortion_bone, distortion_exp , \
                            self.loss_reg.item(), self.beta, self.loss_rev.item(), self.loss_direct,self.loss.item(), time_end -time_start)
            self.logger.info(self.msg)


    def validate_main_process(self):
        with torch.no_grad():
            self.train_main_process()

    def validate_process_images(self):
        super().validate_process_images()
        denoise_bone = self.denoise_bone[:,:,:self.H,:self.W]
        self.denoise255_bone = tensor2image(denoise_bone)

    def validate_cal_metrics(self):
        # calculate metrics
        denoise_psnr = calculate_psnr(self.ori255,self.denoise255)
        denoise_ssim = calculate_ssim(self.ori255,self.denoise255)

        denoise_bone_psnr = calculate_psnr(self.ori255,self.denoise255_bone)
        denoise_bone_ssim = calculate_ssim(self.ori255,self.denoise255_bone)
    
        denoise_exp_psnr = calculate_psnr(self.ori255,self.denoise255_exp)
        denoise_exp_ssim = calculate_ssim(self.ori255,self.denoise255_exp)

        denoise_exp_psnr = calculate_psnr(self.ori255,self.denoise255_exp)
        denoise_exp_ssim = calculate_ssim(self.ori255,self.denoise255_exp)

        denoise_mid_psnr = calculate_psnr(self.ori255,self.denoise255_mid)
        denoise_mid_ssim = calculate_ssim(self.ori255,self.denoise255_mid)
        # collect and append result
        self.curr_validate_result = {'img_name':self.data_name,
                                      'denoise_psnr': denoise_psnr,
                                      'denoise_ssim': denoise_ssim,
                                      'denoise_bone_psnr': denoise_bone_psnr,
                                      'denoise_bone_ssim': denoise_bone_ssim,
                                      'denoise_exp_psnr': denoise_exp_psnr,
                                      'denoise_exp_ssim': denoise_exp_ssim,
                                      'denoise_mid_psnr': denoise_mid_psnr,
                                      'denoise_mid_ssim':denoise_mid_ssim,
                                      }
        self.validate_results.append(self.curr_validate_result)