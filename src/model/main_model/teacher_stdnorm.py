"""BSS with image-level variance matching and invalid-pair skipping."""
import torch
from .teacher import OneTeacher_TwoBranch, OneTeacher_TwoBranch_SIDDRaw

class OneTeacher_TwoBranch_StdNorm(OneTeacher_TwoBranch):
    """Cross-image residual transfer with the target/source standard-deviation ratio."""

    def _variance_ratio(self, source_std, target_std, source_mask, target_mask):
        source_count = source_mask.flatten(1).sum(dim=1)
        target_count = target_mask.flatten(1).sum(dim=1)
        self.valid_pairs = (source_count > 0) & (target_count > 0) & torch.isfinite(source_std) & torch.isfinite(target_std)
        source_std = torch.where(self.valid_pairs, source_std, torch.ones_like(source_std))
        target_std = torch.where(self.valid_pairs, target_std, torch.zeros_like(target_std))
        return (target_std / source_std.clamp(min=0.0001)).view(-1, 1, 1, 1)

    def train_cal_loss(self):

        self.skip_optimization = not self.valid_pairs.any().item()
        if self.skip_optimization:
            self.loss = self.student.new_zeros(())
            return
        keep = self.valid_pairs
        mask = self.high_mask[keep]
        difference = (self.student[keep] - self.teacher_out[keep]) * mask
        self.loss = difference.square().sum() / mask.sum().clamp(min=1)

    def optimize_params(self):
        if self.skip_optimization:
            self.optimizer.zero_grad()
            return
        super().optimize_params()

    def train_main_process(self):
        if self.input.shape[0] < 2:
            raise ValueError('Cross-image re-noising requires batch_size >= 2')
        if self.train_opt.mode == 'v4_stdmask_stdnorm':

            def cal_mask_std(image, mask):
                n, c, h, w = image.shape
                mask = mask.repeat(1, c, 1, 1)
                image = image.view(n, -1)
                mask = mask.view(n, -1)
                mean = torch.sum(image * mask, dim=-1) / torch.sum(mask, dim=-1).clamp(min=0.0001)
                mean = mean.view(n, 1)
                var = torch.sum(((image - mean) * mask) ** 2, dim=-1) / torch.sum(mask, dim=-1).clamp(min=0.0001)
                std = var.clamp(min=0).sqrt()
                return std
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
                std_a = cal_mask_std(teacher1_noise_ranbatch, an_mask)
                std_b = cal_mask_std(teacher1_noise, std_mask)
                ratio = self._variance_ratio(std_a, std_b, an_mask, std_mask)
                renoisy = teacher1_noise_ranbatch * ratio + self.teacher1
                input_mask = an_mask * self.valid_pairs.view(-1, 1, 1, 1)
            self.high_mask = input_mask
            noiser_mixer = torch.where(input_mask.bool(), renoisy, self.input)

            self.student = self.backbone(noiser_mixer)
            self.teacher_out = self.teacher1
        elif self.train_opt.mode == 'v4_stdmask_poinorm_stdnorm':

            def cal_mask_std(image, mask):
                n, c, h, w = image.shape
                mask = mask.repeat(1, c, 1, 1)
                image = image.view(n, -1)
                mask = mask.view(n, -1)
                mean = torch.sum(image * mask, dim=-1) / torch.sum(mask, dim=-1).clamp(min=0.0001)
                mean = mean.view(n, 1)
                var = torch.sum(((image - mean) * mask) ** 2, dim=-1) / torch.sum(mask, dim=-1).clamp(min=0.0001)
                std = var.clamp(min=0).sqrt()
                return std
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
                std_a = cal_mask_std(teacher1_noise_ranbatch, an_mask)
                std_b = cal_mask_std(teacher1_noise, std_mask)
                ratio = self._variance_ratio(std_a, std_b, an_mask, std_mask)
                norm_weight = self.loss_params.norm_weight
                renoisy = teacher1_noise_ranbatch * ratio * self.teacher1.clamp(min=0).sqrt() * norm_weight + self.teacher1
                input_mask = an_mask * self.valid_pairs.view(-1, 1, 1, 1)
            self.high_mask = input_mask
            noiser_mixer = torch.where(input_mask.bool(), renoisy, self.input)

            self.student = self.backbone(noiser_mixer)
            self.teacher_out = self.teacher1
        elif self.train_opt.mode == 'v4_stdmask_poinorm_stdnorm_v2':

            def cal_mask_std(image, mask):
                n, c, h, w = image.shape
                mask = mask.repeat(1, c, 1, 1)
                image = image.view(n, -1)
                mask = mask.view(n, -1)
                mean = torch.sum(image * mask, dim=-1) / torch.sum(mask, dim=-1).clamp(min=0.0001)
                mean = mean.view(n, 1)
                var = torch.sum(((image - mean) * mask) ** 2, dim=-1) / torch.sum(mask, dim=-1).clamp(min=0.0001)
                std = var.clamp(min=0).sqrt()
                return std
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
                std_a = cal_mask_std(teacher1_noise_ranbatch, an_mask)
                std_b = cal_mask_std(teacher1_noise, std_mask)
                ratio = self._variance_ratio(std_a, std_b, an_mask, std_mask)
                renoisy = teacher1_noise_ranbatch * ratio * (self.teacher1.clamp(min=0) + norm_bias).sqrt() + self.teacher1
                input_mask = an_mask * self.valid_pairs.view(-1, 1, 1, 1)
            self.high_mask = input_mask
            noiser_mixer = torch.where(input_mask.bool(), renoisy, self.input)

            self.student = self.backbone(noiser_mixer)
            self.teacher_out = self.teacher1
        else:
            raise ValueError('Unsupported stdnorm mode: {}'.format(self.train_opt.mode))

class OneTeacher_TwoBranch_StdNorm_SIDDRaw(OneTeacher_TwoBranch_StdNorm, OneTeacher_TwoBranch_SIDDRaw):
    """Use the same variance transfer with the existing packed-raw data path."""
