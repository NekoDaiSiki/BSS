import os
import time

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from src.model.main_model.base_model import BaseModel
from src.utils.register import build_from_cfg
from src.utils.builder import MASKER,BACKBONE,NOISER,HEAD
from src.metrics.basic import calculate_psnr_from_tensor,calculate_psnr,calculate_ssim
from src.utils.img_convert import tensor2image
from src.utils.utils import mkdir
from scipy.io import savemat,loadmat
import random

class Noise2Clean(BaseModel):
    def __init__(self, opt):
        super(Noise2Clean,self).__init__(opt)

    def create_models(self):
        self.noise_adder = build_from_cfg(self.sub_models_opt.noiser,NOISER)
        self.backbone = build_from_cfg(self.sub_models_opt.backbone, BACKBONE)
        self.sub_models.update({
            'backbone': self.backbone,
        })

    def train_init(self):
        self.time_start = time.time()
        self.data = self.data/255.0 # normlize input
        # pre process data 
        self.noisyA = self.noise_adder.add_noise(self.data)
        self.input=self.noisyA
        self.target = self.data
    
    def train_main_process(self):
        self.denoise = self.backbone(self.input)
    
    def train_cal_loss(self):
        diff = self.denoise - self.target
        loss_type = self.train_opt.loss_params.loss_type
        if loss_type == 'l1':
            loss = torch.mean(diff.abs())
        elif loss_type == 'l2':
            loss = torch.mean(diff**2)
        else:
            raise NotImplementedError
        self.loss = loss

    def train_logging(self):
        if self.curr_iter%self.train_opt.print_freq ==0:
            self.get_msg()
            self.logger.info(self.msg)

    def get_msg(self):
        lr = self.scheduler.get_last_lr()[0]

        time_start = self.time_start
        time_end = time.time()

        dn = self.denoise
        ori = self.data
        distortion = calculate_psnr_from_tensor(ori,dn)
        self.msg = '{:04d} {:05d} lr={:.2e} distortion={:.6f}, Loss_All={:.6f}, Time={:.4f}'.format(
                self.epoch,self.curr_iter, lr, distortion, self.loss.item(), time_end -time_start)
    def validate_init(self):
        self.data = self.data/255.0
        self.noisy = self.noise_adder.add_noise(self.data)
        n,c,h,w = self.data.shape
        self.H,self.W = h,w
        val_size_h = (h + 31) // 32 * 32
        val_size_w = (w + 31) // 32 * 32
        self.noisy = F.pad(self.noisy,[0,val_size_w-w,0,val_size_h-h],mode='reflect')
        self.input = self.noisy


    def validate_main_process(self):
        with torch.no_grad():
            self.denoise = self.backbone(self.input)

    def validate_process_images(self):
        # unpadding
        denoise = self.denoise[:,:,:self.H,:self.W]
        self.denoise255 = tensor2image(denoise)
        self.ori255 = tensor2image(self.data)
        self.noisy255 = tensor2image(self.noisy[:,:,:self.H,:self.W])

    def validate_cal_metrics(self,repeat,index):
        curr_validate_result = self.get_val_result(repeat,index)
        self.logger.info(curr_validate_result) if self.is_test else print('Not log details in validate')
        self.validate_results.append(curr_validate_result)
    def get_val_result(self,repeat,index):
        # calculate metrics
        denoise_psnr = calculate_psnr(self.ori255,self.denoise255)
        denoise_ssim = calculate_ssim(self.ori255,self.denoise255)
        # collect and append result
        data_name = "{:03d}-{:03d}-{:03d}_{}".format(
                    index,repeat, self.epoch, self.data_name[0])

        curr_validate_result = {'img_name':data_name,
                                      'denoise_psnr': denoise_psnr,
                                      'denoise_ssim': denoise_ssim,
                                      }
        self.logger.info(curr_validate_result)
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
        self.logger.info("epoch:{},dn:{:.6f}/{:.6f}".format(
                    self.epoch, avg_results['denoise_psnr'], avg_results['denoise_ssim']))

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

class Noise2Clean_SIDDsrgb_SA(Noise2Clean):
    def train_init(self):
        self.time_start = time.time()
        noisy = self.data['noisy']
        gt = self.data['gt']
        self.data = gt
        # self.data = self.data/255.0 # normlize input
        # pre process data 
        # self.noisyA = self.noise_adder.add_noise(self.data)
        self.noisyA = noisy
        self.input=self.noisyA
        self.target = gt

    def get_msg(self):
        lr = self.scheduler.get_last_lr()[0]

        time_start = self.time_start
        time_end = time.time()

        dn = self.denoise/255.0
        ori = self.data/255.0
        distortion = calculate_psnr_from_tensor(ori,dn)
        self.msg = '{:04d} {:05d} lr={:.2e} distortion={:.6f}, Loss_All={:.6f}, Time={:.4f}'.format(
                self.epoch,self.curr_iter, lr, distortion, self.loss.item(), time_end -time_start)

    def validate_init(self):
        self.noisy = self.data['noisy']
        self.data = self.data['gt']
        n,c,h,w = self.data.shape
        self.input = self.noisy        

    def validate_process_images(self):
        # self.denoise = denoise
        denoise = self.denoise/255.0
        data = self.data/255.0
        noisy = self.noisy/255.0
        
        self.denoise_01 = denoise.permute(0,2,3,1).cpu().data.clamp(0,1).numpy().squeeze(0)
        self.ori_01 = data.permute(0,2,3,1).cpu().data.clamp(0,1).numpy().squeeze(0)
        self.noisy_01 = noisy.permute(0,2,3,1).cpu().data.clamp(0,1).numpy().squeeze(0)

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

class Noise2Clean_SIDDRaw(Noise2Clean):
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
        # self.data = self.data/255.0 # normlize input
        # pre process data 
        # self.noisyA = self.noise_adder.add_noise(self.data)
        self.noisyA = self.data
        self.input=self.noisyA
        self.target = self.data

    def validate_init(self):
        # self.data = self.data/255.0
        self.noisy = self.data['noisy']
        self.data = self.data['gt']
        # self.data = self.space_to_depth(self.data,2)

        # self.noisy = self.noise_adder.add_noise(self.data)
        n,c,h,w = self.data.shape
        # self.H,self.W = h,w
        # val_size_h = (h + 31) // 32 * 32
        # val_size_w = (w + 31) // 32 * 32
        # self.noisy = F.pad(self.noisy,[0,val_size_w-w,0,val_size_h-h],mode='reflect')
        self.input = self.space_to_depth(self.noisy,block_size=2)

    def validate_process_images(self):
        # self.denoise = denoise
        denoise = self.depth_to_space(self.denoise,block_size=2)
        
        self.denoise_01 = denoise.permute(0,2,3,1).cpu().data.clamp(0,1).numpy().squeeze(0)
        self.ori_01 = self.data.permute(0,2,3,1).cpu().data.clamp(0,1).numpy().squeeze(0)
        self.noisy_01 = self.noisy.permute(0,2,3,1).cpu().data.clamp(0,1).numpy().squeeze(0)

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

    # def save_results(self,repeat=0,idx=0):
    #     if (self.is_train and self.validate_opt.save_results) or (self.is_test and self.test_opt.save_results):
    #         save_dir = os.path.join(self.train_url,'val_imgs')

    #         save_path = os.path.join(
    #             save_dir,
    #             "{:03d}-{:03d}-{:03d}_dn.png".format(
    #                 idx,repeat, self.epoch))
    #         Image.fromarray(self.denoise255.squeeze()).save(save_path)                
    #         save_path = os.path.join(
    #             save_dir,
    #             "{:03d}-{:03d}-{:03d}_clean.png".format(
    #                 idx,repeat, self.epoch))
    #         Image.fromarray(self.ori255.squeeze()).save(save_path)  

    #         save_path = os.path.join(
    #             save_dir,
    #             "{:03d}-{:03d}-{:03d}_noisy.png".format(
    #                 idx,repeat, self.epoch))
    #         Image.fromarray(self.noisy255.squeeze()).save(save_path) 


    def save_results(self,repeat=0,idx=0):
        if (self.is_train and self.validate_opt.save_results) or (self.is_test and self.test_opt.save_results):
            out = 'val_imgs' if self.is_train else 'test_imgs'
            save_dir = os.path.join(self.train_url,out)
            
            save_path = os.path.join(
                save_dir,
                "{:03d}-{:03d}-{:03d}_dn.MAT".format(
                    idx,repeat, self.epoch))

            savemat(save_path, {"x": np.ascontiguousarray(self.denoise_01.squeeze(-1))})

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

class Noise2Clean_SIDDRaw_Train(Noise2Clean_SIDDRaw):
    def train_init(self):
        self.time_start = time.time()
        noisy = self.data['noisy']
        gt = self.data['gt']
        noisy= self.space_to_depth(noisy,2)
        gt= self.space_to_depth(gt,2)
        self.data = gt
        # self.data = self.data/255.0 # normlize input
        # pre process data 
        # self.noisyA = self.noise_adder.add_noise(self.data)
        self.noisyA = noisy
        self.input=self.noisyA
        self.target = gt

class Noise2Noise(Noise2Clean):
    def train_init(self):
        super().train_init()
        self.noisyB = self.noise_adder.add_noise(self.data)
        self.target = self.noisyB
    
class Noise2Noise2(Noise2Clean):
    def create_models(self):
        super().create_models()
        self.noise_adder_2 = build_from_cfg(self.sub_models_opt.noiser_2,NOISER)
    def train_init(self):
        super().train_init()
        self.noisyB = self.noise_adder_2.add_noise(self.data)
        self.target = self.noisyB

    # def train_cal_loss(self):
    #     super().train_cal_loss()
        
    #     if 'w' in self.train_opt.loss_params:
    #         self.loss  = self.loss*self.train_opt.loss_params.w

class Noise2Noise_Identity(Noise2Clean):
    # 为了看恒等映射到底是怎么训出来的 并研究如何避免
    def train_init(self):
        super().train_init()
        self.target = self.input   

class Noise2Noise_Identity_EhenHead(Noise2Noise_Identity):
    # 拿到一个好的代理任务 加一个head直接学

    def create_models(self):
        super().create_models()
        self.enhance_head = build_from_cfg(self.sub_models_opt.enhance_head,BACKBONE)
        self.sub_models.update({
            'enhance_head': self.enhance_head,
        })
    def train_main_process(self):
        self.denoise_exp = self.backbone(self.input).detach()  # 可以选择不优化
        kmap = self.enhance_head(torch.cat([self.input,self.denoise_exp],dim=1))
        self.kmap = kmap.exp()+1
        self.noise_exp = self.target - self.denoise_exp
        self.noise_exp_k = self.kmap * self.noise_exp
        self.denoise_exp_k = self.target - self.noise_exp_k

    def train_cal_loss(self):
        loss_params = self.train_opt.loss_params
        alpha,beta = loss_params.alpha,loss_params.beta
        corr = self.denoise_exp_k * self.noise_exp_k.detach()
        self.corr_loss = beta*torch.mean(corr).abs() 
        diff = self.denoise_exp_k - self.denoise_exp
        self.mse_loss = alpha*torch.mean(diff**2)
        self.loss = self.corr_loss + self.mse_loss


    def train_logging(self):
        if self.curr_iter%self.train_opt.print_freq ==0:
            lr = self.scheduler.get_last_lr()[0]

            time_start = self.time_start
            time_end = time.time()

            dn_exp = self.denoise_exp
            dn_exp_k = self.denoise_exp_k
            ori = self.data
            distortion = calculate_psnr_from_tensor(ori,dn_exp)
            distortion_k = calculate_psnr_from_tensor(ori,dn_exp_k)
            self.msg = '{:04d} {:05d} lr={:.2e} distortion={:.6f} distortion_k={:.6f} corr_loss={:.6f} mse_loss={:.6f} Loss_All={:.6f}, Time={:.4f}'.format(
                    self.epoch,self.curr_iter, lr, distortion, distortion_k,self.corr_loss.item(),self.mse_loss.item(), self.loss.item(), time_end -time_start)
            self.logger.info(self.msg)

    def validate_main_process(self):
        with torch.no_grad():
            denoise = self.backbone(self.input)
            self.denoise = self.enhance_head(denoise)


class SiameseContrast(Noise2Clean):
    def create_models(self):
        super().create_models()
        self.anchor = build_from_cfg(self.sub_models_opt.backbone, BACKBONE)
        self.sub_models.update({
            'anchor': self.anchor,
        })   
    def train_main_process(self):
        self.denoise1 = self.backbone(self.input)
        self.denoise12 = self.backbone(torch.flip(self.input,dims=[2]))
        self.denoise13 = self.backbone(torch.flip(self.input,dims=[3]))
        self.denoise1 = 1/3*(self.denoise1+torch.flip(self.denoise12,dims=[2])+torch.flip(self.denoise13,dims=[3]))
        self.noise1 = self.noisyA - self.denoise1

        self.noise1_rev = torch.flip(self.noise1,dims=[0])
        

        input2 = (self.denoise1 + self.noise1_rev).detach()
        self.denoise2 = self.backbone(input2)
        self.noise2 = self.noisyA - self.denoise2
        with torch.no_grad():
            self.anchor_out = self.anchor(self.input)


    def train_cal_loss(self):
        
        diff1 = self.denoise1 - self.noisyA
        diff2 = self.denoise2 - self.anchor_out
        loss1 = 0*torch.mean(diff1**2)
        loss2 = 0*torch.mean(diff2**2)
        self.loss = loss1 + loss2

    # def optimize_params(self):
    #     pass
    def get_msg(self):
        lr = self.scheduler.get_last_lr()[0]

        time_start = self.time_start
        time_end = time.time()

        dn1 = self.denoise1
        dn2 = self.denoise2
        ori = self.data
        distortion1 = calculate_psnr_from_tensor(ori,dn1)
        distortion2 = calculate_psnr_from_tensor(ori,dn2)
        anchor = calculate_psnr_from_tensor(ori,self.anchor_out)
        self.msg = '{:04d} {:05d} lr={:.2e} distortion1={:.6f},distortion2={:.6f},anchor={:.6f}, Loss_All={:.6f}, Time={:.4f}'.format(
                self.epoch,self.curr_iter, lr, distortion1, distortion2,anchor, self.loss.item(), time_end -time_start)
        
class Exchange(SiameseContrast):
    def __init__(self, opt):
        super().__init__(opt)
        self.m_list = self.train_opt.m_list #0.999
        self.stage_list = self.train_opt.stage_list
    @torch.no_grad()
    def momentum_update_key_encoder(self,model,model_m):
        m = self.m_list[self.curr_stage]
        if m==-1:
            return None
        for param, param_m in zip(model.parameters(), model_m.parameters()):
            param_m.data = param_m.data * m + param.data * (1. - m)
        self.m = m
    def train_epoch_init(self, train_logger, epoch):
        super().train_epoch_init(train_logger, epoch)
        for i in range(len(self.stage_list)):
            if self.epoch>=self.stage_list[i]:
                self.curr_stage = i

    def train_main_process(self):
        with torch.no_grad():
            self.anchor_out = self.anchor(self.input)
            aout_noise = self.input - self.anchor_out
            an_list = aout_noise.chunk(4,dim=0)
            id_list = list(range(len(an_list)))
            random.shuffle(id_list)
            # print(id_list)
            an_shuffle_list = []
            for i in id_list:
                an_shuffle_list.append(an_list[i])
            aout_noise_re1 = torch.cat(an_shuffle_list,dim=0)
            # aout_noise_re2 = torch.cat([an_4,an_1,an_2,an_3],dim=0)
            self.denoise1 = self.backbone(self.input)

        self.denoise2 = self.backbone(aout_noise_re1+self.anchor_out)

    def train_cal_loss(self):
        
        diff1 = self.denoise1 - self.input
        diff2 = self.denoise2 - self.anchor_out
        loss1 = 0*torch.mean(diff1**2)
        loss2 = torch.mean(diff2**2)
        self.loss = loss2

    def optimize_params(self):
        super().optimize_params()
        self.momentum_update_key_encoder(self.backbone,self.anchor)

    def validate_main_process(self):
        with torch.no_grad():
            self.anchor_out = self.anchor(self.input)
            self.denoise1 = self.backbone(self.input)

    def validate_process_images(self):
        # unpadding
        denoise1 = self.denoise1[:,:,:self.H,:self.W]
        anchor = self.anchor_out[:,:,:self.H,:self.W]
        self.denoise255 = tensor2image(denoise1)
        self.anchor255 = tensor2image(anchor)
        self.ori255 = tensor2image(self.data)
        self.noisy255 = tensor2image(self.noisy[:,:,:self.H,:self.W])

    def get_val_result(self):
        # calculate metrics
        denoise_psnr = calculate_psnr(self.ori255,self.denoise255)
        denoise_ssim = calculate_ssim(self.ori255,self.denoise255)

        anchor_psnr = calculate_psnr(self.ori255,self.anchor255)
        anchor_ssim = calculate_ssim(self.ori255,self.anchor255)
        # collect and append result
        curr_validate_result = {'img_name':self.data_name,
                                      'denoise_psnr': denoise_psnr,
                                      'denoise_ssim': denoise_ssim,
                                      'anchor_psnr': anchor_psnr,
                                      'anchor_ssim': anchor_ssim,
                                      }
        print(curr_validate_result)
        return curr_validate_result
    
    def logger_val_summary(self,avg_results):
        self.logger.info("epoch:{},dn:{:.6f}/{:.6f}, anchor:{:.6f}/{:.6f}".format(
                    self.epoch, avg_results['denoise_psnr'], avg_results['denoise_ssim'],
                                avg_results['anchor_psnr'], avg_results['anchor_ssim']))


class OneTeacher(Noise2Clean):
    def create_models(self):
        super().create_models()
        self.net_teacher1 = build_from_cfg(self.sub_models_opt.teacher1, BACKBONE)
        self.sub_models.update({
            'teacher1': self.net_teacher1,
        })   

    def rotate(self, x, flag,rev=False):
        angle_list = [0,90,180,270]
        angle = angle_list[flag] # 0,1,2,3        
        if rev:
            angle = (360-angle)%360

        if angle == 0:
            return x
        elif angle == 90:
            return torch.rot90(x, k=1, dims=(3, 2))
        elif angle == 180:
            return torch.rot90(x, k=2, dims=(3, 2))
        elif angle == 270:
            return torch.rot90(x, k=3, dims=(3, 2))

            
    def flip(self,x, flag):
        # falg: 0,1,2
        if flag ==0:
            return torch.flip(x,dims=[2])
        elif flag ==1:
            return torch.flip(x,dims=[3])
        elif flag ==2:
            return torch.flip(x,dims=[2,3])

    
    def train_main_process(self):

        if self.train_opt.mode == 'mode0':
            with torch.no_grad():
                self.teacher1 = self.net_teacher1(self.input)
            self.student = self.backbone(self.input)
            self.teacher_out = self.teacher1

        elif self.train_opt.mode == 'rot_and_flip':
            n,c,h,w = self.input.shape
            rot_flag = random.choices([0,1,2,3],k=n)
            flip_flag = random.choices([0,1,2],k=n)
            chunk_input = torch.chunk(self.input,chunks=n,dim=0)
            aug_input = torch.cat([self.flip(self.rotate(chunk_input[i],rot_flag[i],rev=False),flip_flag[i]) for i in range(n)],dim=0)
            with torch.no_grad():
                self.teacher1 = self.net_teacher1(self.input)
                teacher_out = self.net_teacher1(aug_input)
            chunk_teacher_out = torch.chunk(teacher_out,chunks=n,dim=0)
            teacher_out_rev = torch.cat([self.rotate(self.flip(chunk_teacher_out[i],flip_flag[i]),rot_flag[i],rev=True) for i in range(n)],dim=0)
            self.teacher_out = teacher_out_rev
            self.student = self.backbone(self.input)

        elif self.train_opt.mode == 'rot_and_flip_batch':
            n,c,h,w = self.input.shape
            rot_flag = random.choices([0,1,2,3],k=1)
            flip_flag = random.choices([0,1,2],k=1)
            aug_input = self.flip(self.rotate(self.input,rot_flag[0],rev=False),flip_flag[0])
            with torch.no_grad():
                self.teacher1 = self.net_teacher1(self.input)
                teacher_out = self.net_teacher1(aug_input)
            teacher_out_rev = self.rotate(self.flip(teacher_out,flip_flag[0]),rot_flag[0],rev=True)
            self.teacher_out = teacher_out_rev
            self.student = self.backbone(self.input)
        
        elif self.train_opt.mode == 'random_exchange_batch':
            with torch.no_grad():
                self.teacher1 = self.net_teacher1(self.input)
                teacher1_noise = self.input - self.teacher1
                n,c,h,w = self.input.shape
                an_list = teacher1_noise.chunk(n,dim=0)
                id_list = list(range(len(an_list)))
                random.shuffle(id_list)
                # print(id_list)
                an_shuffle_list = []
                for i in id_list:
                    an_shuffle_list.append(an_list[i])
                teacher1_noise_ranbatch = torch.cat(an_shuffle_list,dim=0)
            self.student = self.backbone(teacher1_noise_ranbatch+self.teacher1)
            self.teacher_out = self.teacher1

        elif self.train_opt.mode == 'shift_batch':
            with torch.no_grad():
                self.teacher1 = self.net_teacher1(self.input)
                teacher1_noise = self.input - self.teacher1
                n,c,h,w = self.input.shape
                an_list = teacher1_noise.chunk(n,dim=0)
                if n>1:
                    shift_num = random.randint(1,n-1) 
                    an_shift_list = an_list[shift_num:] + an_list[:shift_num]
                else:
                    an_shift_list = an_list
                teacher1_noise_ranbatch = torch.cat(an_shift_list,dim=0)
            self.student = self.backbone(teacher1_noise_ranbatch+self.teacher1)
            self.teacher_out = self.teacher1
        
        elif self.train_opt.mode == 'cross_exchange_w2':
            with torch.no_grad():
                self.teacher1 = self.net_teacher1(self.input)
                self.student_exp = self.backbone(self.input)
                teacher1_noise = self.input - self.teacher1  
                n,c,h,w = self.input.shape
                teacher1_noise_cross = teacher1_noise.view(n,c,h//2,2,w//2,2).permute(0,1,2,4,3,5).reshape(n,c,h//2,w//2,-1).chunk(4,dim=4)
                teacher1_noise_cross = torch.cat(teacher1_noise_cross[::-1],dim=4)
                teacher1_noise_cross = teacher1_noise_cross.view(n,c,h//2,w//2,2,2).permute(0,1,2,4,3,5).reshape(n,c,h,w)
            self.student = self.backbone(teacher1_noise_cross+self.teacher1)
            self.teacher_out = self.teacher1

        elif self.train_opt.mode == 'shift_w2':
            with torch.no_grad():
                self.teacher1 = self.net_teacher1(self.input)
                teacher1_noise = self.input - self.teacher1  
                n,c,h,w = self.input.shape
                teacher1_noise_cross = teacher1_noise.view(n,c,h//2,2,w//2,2).permute(0,1,2,4,3,5).reshape(n,c,h//2,w//2,-1)
                shift_num = random.randint(1,3) 
                teacher1_noise_cross = torch.cat([teacher1_noise_cross[:,:,:,:,shift_num:],teacher1_noise_cross[:,:,:,:,:shift_num]],dim=4)
                teacher1_noise_cross = teacher1_noise_cross.view(n,c,h//2,w//2,2,2).permute(0,1,2,4,3,5).reshape(n,c,h,w)
            self.student = self.backbone(teacher1_noise_cross+self.teacher1)
            self.teacher_out = self.teacher1

        elif self.train_opt.mode == 'shift_w2_n':
            with torch.no_grad():
                self.teacher1 = self.net_teacher1(self.input)
                self.student_exp = self.backbone(self.input)
                teacher1_noise = self.input - self.teacher1  
                n,c,h,w = self.input.shape
                teacher1_noise_cross = teacher1_noise.view(n,c,h//2,2,w//2,2).permute(0,1,2,4,3,5).reshape(n,c,h//2,w//2,-1)
                shift_num = random.choices([1,2,3],k=2) 
                teacher1_noise_cross1 = torch.cat([teacher1_noise_cross[:,:,:,:,shift_num[0]:],teacher1_noise_cross[:,:,:,:,:shift_num[0]]],dim=4)
                teacher1_noise_cross1 = teacher1_noise_cross1.view(n,c,h//2,w//2,2,2).permute(0,1,2,4,3,5).reshape(n,c,h,w)

                teacher1_noise_cross2 = torch.cat([teacher1_noise_cross[:,:,:,:,shift_num[1]:],teacher1_noise_cross[:,:,:,:,:shift_num[1]]],dim=4)
                teacher1_noise_cross2 = teacher1_noise_cross2.view(n,c,h//2,w//2,2,2).permute(0,1,2,4,3,5).reshape(n,c,h,w)              
            self.student = self.backbone(teacher1_noise_cross2+self.teacher1)
            self.teacher_out = self.teacher1 

        elif self.train_opt.mode == 'shift_w4':
            with torch.no_grad():
                self.teacher1 = self.net_teacher1(self.input)
                self.student_exp = self.backbone(self.input)
                teacher1_noise = self.input - self.teacher1  
                n,c,h,w = self.input.shape
                teacher1_noise_cross = teacher1_noise.view(n,c,h//4,4,w//4,4).permute(0,1,2,4,3,5).reshape(n,c,h//4,w//4,-1)
                shift_num = random.randint(1,15) 
                teacher1_noise_cross = torch.cat([teacher1_noise_cross[:,:,:,:,shift_num:],teacher1_noise_cross[:,:,:,:,:shift_num]],dim=4)
                teacher1_noise_cross = teacher1_noise_cross.view(n,c,h//4,w//4,4,4).permute(0,1,2,4,3,5).reshape(n,c,h,w)
            self.student = self.backbone(teacher1_noise_cross+self.teacher1)
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


    def get_val_result(self):
        # calculate metrics
        teacher1_psnr = calculate_psnr(self.ori255,self.teacher1_255)
        teacher1_ssim = calculate_ssim(self.ori255,self.teacher1_255)

        student_psnr = calculate_psnr(self.ori255,self.student_255)
        student_ssim = calculate_ssim(self.ori255,self.student_255)
        # collect and append result
        curr_validate_result = {'img_name':self.data_name,
                                      'teacher1_psnr': teacher1_psnr,
                                      'teacher1_ssim': teacher1_ssim,
                                      'student_psnr': student_psnr,
                                      'student_ssim': student_ssim,
                                      }
        print(curr_validate_result)
        return curr_validate_result
    
    def logger_val_summary(self,avg_results):
        self.logger.info("epoch:{},teacher1:{:.6f}/{:.6f}, student:{:.6f}/{:.6f}".format(
                    self.epoch, avg_results['teacher1_psnr'], avg_results['teacher1_ssim'],
                                avg_results['student_psnr'], avg_results['student_ssim'],
                                ))




class TwoTeacher(Noise2Clean):
    def create_models(self):
        super().create_models()
        self.net_teacher1 = build_from_cfg(self.sub_models_opt.teacher1, BACKBONE)
        self.net_teacher2 = build_from_cfg(self.sub_models_opt.teacher2, BACKBONE)
        self.sub_models.update({
            'teacher1': self.net_teacher1,
            'teacher2': self.net_teacher2,
        })   

    def train_main_process(self):
        with torch.no_grad():
            self.teacher1 = self.net_teacher1(self.input)
            self.teacher2 = self.net_teacher2(self.input)

        if self.train_opt.mode == 'mode0':
            self.student = self.backbone(self.input)
            self.teacher_out = 0.5*(self.teacher1+self.teacher2)

        elif self.train_opt.mode == 'random_mix':
            self.student = self.backbone(self.input)
            mix_weight = torch.rand([1],device=self.device)
            self.teacher_out = mix_weight*self.teacher1+(1-mix_weight)*self.teacher2

    def validate_main_process(self):
        with torch.no_grad():
            self.train_main_process()

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
        teacher2 = calculate_psnr_from_tensor(ori,self.teacher2)
        student = calculate_psnr_from_tensor(ori,self.student)
        self.msg = '{:04d} {:05d} lr={:.2e} teacher1={:.6f}, teacher2={:.6f}, student={:.6f}, Loss_All={:.6f}, Time={:.4f}'.format(
                self.epoch,self.curr_iter, lr, teacher1, teacher2, student, self.loss.item(), time_end -time_start)

    def validate_process_images(self):
        # unpadding
        teacher1 = self.teacher1[:,:,:self.H,:self.W]
        teacher2 = self.teacher2[:,:,:self.H,:self.W]
        student = self.student[:,:,:self.H,:self.W]

        self.teacher1_255 = tensor2image(teacher1)
        self.teacher2_255 = tensor2image(teacher2)
        self.student_255 = tensor2image(student)
        self.ori255 = tensor2image(self.data)
        self.noisy255 = tensor2image(self.noisy[:,:,:self.H,:self.W])


    def get_val_result(self):
        # calculate metrics
        teacher1_psnr = calculate_psnr(self.ori255,self.teacher1_255)
        teacher1_ssim = calculate_ssim(self.ori255,self.teacher1_255)

        teacher2_psnr = calculate_psnr(self.ori255,self.teacher2_255)
        teacher2_ssim = calculate_ssim(self.ori255,self.teacher2_255)

        student_psnr = calculate_psnr(self.ori255,self.student_255)
        student_ssim = calculate_ssim(self.ori255,self.student_255)
        # collect and append result
        curr_validate_result = {'img_name':self.data_name,
                                      'teacher1_psnr': teacher1_psnr,
                                      'teacher1_ssim': teacher1_ssim,
                                      'teacher2_psnr': teacher2_psnr,
                                      'teacher2_ssim': teacher2_ssim,
                                      'student_psnr': student_psnr,
                                      'student_ssim': student_ssim,
                                      }
        print(curr_validate_result)
        return curr_validate_result
    
    def logger_val_summary(self,avg_results):
        self.logger.info("epoch:{},teacher1:{:.6f}/{:.6f}, teacher2:{:.6f}/{:.6f}, student:{:.6f}/{:.6f}".format(
                    self.epoch, avg_results['teacher1_psnr'], avg_results['teacher1_ssim'],
                                avg_results['teacher2_psnr'], avg_results['teacher2_ssim'],
                                avg_results['student_psnr'], avg_results['student_ssim'],
                                ))