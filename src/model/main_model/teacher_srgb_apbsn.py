"""SIDD sRGB BSS training with an AP-BSN separator and student."""
import os
import time
import numpy as np
from scipy.io import savemat
from src.metrics.basic import calculate_psnr, calculate_psnr_from_tensor, calculate_ssim, calculate_ssim_y
from src.utils.img_convert import tensor2image
from .denoising_base import Noise2Clean_SIDDsrgb_SA
from .teacher import OneTeacher_TwoBranch

class OneTeacher_TwoBranch_SIDDsrgb_SA(OneTeacher_TwoBranch, Noise2Clean_SIDDsrgb_SA):

    def train_main_process(self):
        super().train_main_process()
        self.student = self.student / 255.0
        self.teacher_out = self.teacher_out / 255.0

    def generate_std_mask(self, input, thres, ws=7):
        input = input / 255.0
        return super().generate_std_mask(input, thres, ws)

    def validate_process_images(self):
        teacher1 = self.teacher1 / 255.0
        student = self.student / 255.0
        data = self.data / 255.0
        noisy = self.noisy / 255.0
        self.teacher1_01 = teacher1.permute(0, 2, 3, 1).cpu().data.clamp(0, 1).numpy().squeeze(0)
        self.student_01 = student.permute(0, 2, 3, 1).cpu().data.clamp(0, 1).numpy().squeeze(0)
        self.ori_01 = data.permute(0, 2, 3, 1).cpu().data.clamp(0, 1).numpy().squeeze(0)
        self.noisy_01 = noisy.permute(0, 2, 3, 1).cpu().data.clamp(0, 1).numpy().squeeze(0)
        self.teacher1_255 = tensor2image(teacher1)
        self.student_255 = tensor2image(student)
        self.ori255 = tensor2image(self.data)
        self.noisy255 = tensor2image(self.noisy)
        self.denoise255 = self.student_255

    def get_val_result(self, index, repeat):
        opt = self.test_opt if self.is_test else self.validate_opt
        quantize = getattr(opt, 'quantize_metric', False)
        ssim_mode = getattr(opt, 'ssim_mode', 'y_channel')
        _ssim_fn = calculate_ssim if ssim_mode == 'rgb' else calculate_ssim_y
        if quantize:
            ori_q = np.clip(np.round(self.ori_01 * 255), 0, 255).astype(np.uint8).astype(np.float32)
            t1_q = np.clip(np.round(self.teacher1_01 * 255), 0, 255).astype(np.uint8).astype(np.float32)
            stu_q = np.clip(np.round(self.student_01 * 255), 0, 255).astype(np.uint8).astype(np.float32)
            teacher1_psnr = calculate_psnr(ori_q / 255, t1_q / 255, 1)
            teacher1_ssim = _ssim_fn(ori_q, t1_q)
            student_psnr = calculate_psnr(ori_q / 255, stu_q / 255, 1)
            student_ssim = _ssim_fn(ori_q, stu_q)
        else:
            teacher1_psnr = calculate_psnr(self.ori_01.astype(np.float32), self.teacher1_01.astype(np.float32), 1)
            teacher1_ssim = _ssim_fn(self.ori_01.astype(np.float32) * 255, self.teacher1_01.astype(np.float32) * 255)
            student_psnr = calculate_psnr(self.ori_01.astype(np.float32), self.student_01.astype(np.float32), 1)
            student_ssim = _ssim_fn(self.ori_01.astype(np.float32) * 255, self.student_01.astype(np.float32) * 255)
        curr_validate_result = {'img_name': self.data_name, 'teacher1_psnr': teacher1_psnr, 'teacher1_ssim': teacher1_ssim, 'student_psnr': student_psnr, 'student_ssim': student_ssim}
        print(curr_validate_result)
        return curr_validate_result

    def save_results(self, repeat=0, idx=0):
        if self.is_train and self.validate_opt.save_results or (self.is_test and self.test_opt.save_results):
            out = 'val_imgs' if self.is_train else 'test_imgs'
            save_dir = os.path.join(self.train_url, out)
            save_path = os.path.join(save_dir, '{:03d}-{:03d}-{:03d}_dn.MAT'.format(idx, repeat, self.epoch))
            savemat(save_path, {'x': np.ascontiguousarray(self.student_01)})
            save_path = os.path.join(save_dir, '{:03d}-{:03d}-{:03d}_clean.MAT'.format(idx, repeat, self.epoch))
            savemat(save_path, {'x': np.ascontiguousarray(self.ori_01)})
            save_path = os.path.join(save_dir, '{:03d}-{:03d}-{:03d}_noisy.MAT'.format(idx, repeat, self.epoch))
            savemat(save_path, {'x': np.ascontiguousarray(self.noisy_01)})

    def get_msg(self):
        lr = self.scheduler.get_last_lr()[0]
        time_start = self.time_start
        time_end = time.time()
        ori = self.data / 255.0
        teacher1 = calculate_psnr_from_tensor(ori, self.teacher1 / 255.0)
        student = calculate_psnr_from_tensor(ori, self.student)
        student_exp = calculate_psnr_from_tensor(ori, self.student_exp / 255.0)
        self.msg = '{:04d} {:05d} lr={:.2e} teacher1={:.6f}, student={:.6f}, student_exp={:.6f}, Loss_All={:.6f}, Time={:.4f}'.format(self.epoch, self.curr_iter, lr, teacher1, student, student_exp, self.loss.item(), time_end - time_start)
