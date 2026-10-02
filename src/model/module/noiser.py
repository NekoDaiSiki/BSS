"""Noise generators used by the maintained NSC-BSM and BSS configurations."""

import torch

from src.utils.builder import NOISER


def _torch_version_tuple(version):
    numbers = []
    for part in version.split('+')[0].split('.')[:2]:
        digits = ''.join(char for char in part if char.isdigit())
        numbers.append(int(digits) if digits else 0)
    return tuple((numbers + [0, 0])[:2])


# PyTorch 1.8 broadcasts std onto out; newer releases resize out to std's shape.
_USE_LEGACY_NORMAL_OUT = _torch_version_tuple(torch.__version__) < (1, 10)


def _gaussian_noise(x, std):
    if _USE_LEGACY_NORMAL_OUT:
        noise = torch.empty_like(x)
        torch.normal(mean=0.0, std=std, out=noise)
        return noise
    return torch.randn_like(x) * std


@NOISER.register_module()
class GauPos_Noiser:
    """Fixed Gaussian or Poisson noise for the synthetic training configurations."""

    def __init__(self, style):
        if '_' in style:
            raise ValueError('Expected a fixed noise level: {}'.format(style))
        if style.startswith('gauss'):
            self.style = 'gauss'
            self.level = float(style[5:]) / 255.0
        elif style.startswith('poisson'):
            self.style = 'poisson'
            self.level = float(style[7:])
        else:
            raise ValueError('Unknown noise style: {}'.format(style))
        if self.level < 0 or (self.style == 'poisson' and self.level == 0):
            raise ValueError('Invalid noise level: {}'.format(style))

    def add_noise(self, x):
        if self.style == 'gauss':
            std = x.new_full((x.shape[0], 1, 1, 1), self.level)
            return x + _gaussian_noise(x, std)
        return torch.poisson(self.level * x) / self.level
