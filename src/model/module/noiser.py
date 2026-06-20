import numpy as np
import torch

from src.utils.register import Registry
from src.utils.builder import NOISER

@NOISER.register_module()
class GauPos_Noiser(object):
    def __init__(self, style):
        print('[NOTE]: Noise add type: {}'.format(style))
        if style.startswith('gauss'):
            self.params = [
                float(p) / 255.0 for p in style.replace('gauss', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "gauss_fix"
            elif len(self.params) == 2:
                self.style = "gauss_range"
        elif style.startswith('poisson'):
            self.params = [
                float(p) for p in style.replace('poisson', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "poisson_fix"
            elif len(self.params) == 2:
                self.style = "poisson_range"

        elif style.startswith('Poisson'):
            self.params = [
                float(p) for p in style.replace('Poisson', '').split('_')
            ]
            if len(self.params) == 1:
                self.style = "poissonvF_fix"
            elif len(self.params) == 2:
                self.style = "poissonvF_range"
        # self.generater = generater()
        # self.get_generator = self.generater.get_generator
    def add_noise(self, x):
        '''
        input range [0.,1.], tensor
        '''
        shape = x.shape
        if self.style == "gauss_fix":
            std = self.params[0]
            std = std * torch.ones((shape[0], 1, 1, 1), device=x.device)
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0.0,
                         std=std,
                         out=noise)
            return x + noise
        elif self.style == "gauss_range":
            min_std, max_std = self.params
            std = torch.rand(size=(shape[0], 1, 1, 1),
                             device=x.device) * (max_std - min_std) + min_std
            noise = torch.cuda.FloatTensor(shape, device=x.device)
            torch.normal(mean=0, std=std, out=noise)
            return x + noise
        elif self.style == "poisson_fix":
            lam = self.params[0]
            lam = lam * torch.ones((shape[0], 1, 1, 1), device=x.device)
            noised = torch.poisson(lam * x, ) / lam
            return noised
        elif self.style == "poisson_range":
            min_lam, max_lam = self.params
            lam = torch.rand(size=(shape[0], 1, 1, 1),
                             device=x.device) * (max_lam - min_lam) + min_lam
            noised = torch.poisson(lam * x) / lam

        elif self.style == "poissonvF_fix":
            x = x *255
            lam = self.params[0]
            lam = lam * torch.ones((shape[0], 1, 1, 1), device=x.device)
            lam = lam/255.0
            noised = torch.poisson(lam * x, ) / lam
            noised = noised/255
            return noised
        
        else:
            raise NotImplementedError

    # def add_valid_noise(self, x):
    #     shape = x.shape
    #     if self.style == "gauss_fix":
    #         std = self.params[0]
    #         return np.array(x + np.random.normal(size=shape) * std,
    #                         dtype=np.float32)
    #     elif self.style == "gauss_range":
    #         min_std, max_std = self.params
    #         std = np.random.uniform(low=min_std, high=max_std, size=(1, 1, 1))
    #         return np.array(x + np.random.normal(size=shape) * std,
    #                         dtype=np.float32)
    #     elif self.style == "poisson_fix":
    #         lam = self.params[0]
    #         return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)
    #     elif self.style == "poisson_range":
    #         min_lam, max_lam = self.params
    #         lam = np.random.uniform(low=min_lam, high=max_lam, size=(1, 1, 1))
    #         return np.array(np.random.poisson(lam * x) / lam, dtype=np.float32)


def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)