import time

import torch

from src.model.main_model.base_model import BaseModel
from src.utils.register import build_from_cfg
from src.utils.builder import MASKER, BACKBONE, NOISER

class BlindSpotModel(BaseModel):

    def __init__(self, opt):
        super(BlindSpotModel,self).__init__(opt)
        self.mask = None

    def create_models(self):
        # masker and noiser are not nn.Module
        if 'noiser' in self.sub_models_opt:
            self.train_noise_adder = build_from_cfg(self.sub_models_opt.noiser, NOISER)
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

    def validate_cal_metrics(self,repeat,index):
        curr_validate_result = self.get_val_result(repeat,index)
        self.logger.info(curr_validate_result) if self.is_test else print(curr_validate_result)
        self.validate_results.append(curr_validate_result)

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


class B2UB_Model(BlindSpotModel):

    def train_main_process(self):
        n,c,h,w = self.noisy.shape
        denoise = self.backbone(self.input)
        self.denoise = (denoise*self.mask).view(n, -1, c, h, w).sum(dim=1)

        with torch.no_grad():
            self.exp_denoise = self.backbone(self.noisy)
