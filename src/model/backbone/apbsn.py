import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.builder import BACKBONE

def pixel_shuffle_down_sampling(x: torch.Tensor, f: int, pad: int=0, pad_value: float=0.0):
    if len(x.shape) == 3:
        c, w, h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0:
            unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(c, f, f, w // f + 2 * pad, h // f + 2 * pad).permute(0, 1, 3, 2, 4).reshape(c, w + 2 * f * pad, h + 2 * f * pad)
    b, c, w, h = x.shape
    unshuffled = F.pixel_unshuffle(x, f)
    if pad != 0:
        unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
    return unshuffled.view(b, c, f, f, w // f + 2 * pad, h // f + 2 * pad).permute(0, 1, 2, 4, 3, 5).reshape(b, c, w + 2 * f * pad, h + 2 * f * pad)

def pixel_shuffle_up_sampling(x: torch.Tensor, f: int, pad: int=0):
    if len(x.shape) == 3:
        c, w, h = x.shape
        before_shuffle = x.view(c, f, w // f, f, h // f).permute(0, 1, 3, 2, 4).reshape(c * f * f, w // f, h // f)
        if pad != 0:
            before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)
    b, c, w, h = x.shape
    before_shuffle = x.view(b, c, f, w // f, f, h // f).permute(0, 1, 2, 4, 3, 5).reshape(b, c * f * f, w // f, h // f)
    if pad != 0:
        before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
    return F.pixel_shuffle(before_shuffle, f)

class CentralMaskedConv2d(nn.Conv2d):

    def __init__(self, *args, blindspot=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.blindspot = blindspot
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2] = 0

    def forward(self, x):
        if self.blindspot:
            self.weight.data *= self.mask
        return super().forward(x)

class DCl(nn.Module):

    def __init__(self, stride, in_ch):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride), nn.ReLU(inplace=True), nn.Conv2d(in_ch, in_ch, kernel_size=1))

    def forward(self, x):
        return x + self.body(x)

class DC_branchl(nn.Module):

    def __init__(self, stride, in_ch, num_module):
        super().__init__()
        layers = [CentralMaskedConv2d(in_ch, in_ch, kernel_size=2 * stride - 1, stride=1, padding=stride - 1), nn.ReLU(inplace=True), nn.Conv2d(in_ch, in_ch, kernel_size=1), nn.ReLU(inplace=True), nn.Conv2d(in_ch, in_ch, kernel_size=1), nn.ReLU(inplace=True)]
        layers += [DCl(stride, in_ch) for _ in range(num_module)]
        layers += [nn.Conv2d(in_ch, in_ch, kernel_size=1), nn.ReLU(inplace=True)]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)

class DBSNl(nn.Module):

    def __init__(self, in_ch=3, out_ch=3, base_ch=128, num_module=9):
        super().__init__()
        assert base_ch % 2 == 0, 'base channel should be divided with 2'
        self.head = nn.Sequential(nn.Conv2d(in_ch, base_ch, kernel_size=1), nn.ReLU(inplace=True))
        self.branch1 = DC_branchl(2, base_ch, num_module)
        self.branch2 = DC_branchl(3, base_ch, num_module)
        self.tail = nn.Sequential(nn.Conv2d(base_ch * 2, base_ch, kernel_size=1), nn.ReLU(inplace=True), nn.Conv2d(base_ch, base_ch // 2, kernel_size=1), nn.ReLU(inplace=True), nn.Conv2d(base_ch // 2, base_ch // 2, kernel_size=1), nn.ReLU(inplace=True), nn.Conv2d(base_ch // 2, out_ch, kernel_size=1))

    def forward(self, x):
        x = self.head(x)
        br1 = self.branch1(x)
        br2 = self.branch2(x)
        x = torch.cat([br1, br2], dim=1)
        return self.tail(x)

class APBSNCore(nn.Module):

    def __init__(self, pd_a=5, pd_b=2, pd_pad=2, R3=False, bsn='DBSNl', in_ch=3, bsn_base_ch=128, bsn_num_module=9):
        super().__init__()
        if R3:
            raise ValueError('The AP-BSN BSS configuration requires R3=False')
        self.pd_a = pd_a
        self.pd_b = pd_b
        self.pd_pad = pd_pad
        self.R3 = R3
        if bsn == 'DBSNl':
            self.bsn = DBSNl(in_ch, in_ch, bsn_base_ch, bsn_num_module)
        else:
            raise ValueError('Expected bsn=DBSNl, got {}'.format(bsn))

    def forward_pd(self, img, pd=None):
        if pd is None:
            pd = self.pd_a
        h, w = img.shape[-2:]
        if pd > 1:
            pad_h = (pd - h % pd) % pd
            pad_w = (pd - w % pd) % pd
            if pad_h != 0 or pad_w != 0:
                img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
            pd_img = pixel_shuffle_down_sampling(img, f=pd, pad=self.pd_pad)
        else:
            p = self.pd_pad
            pd_img = F.pad(img, (p, p, p, p))
        pd_img_denoised = self.bsn(pd_img)
        if pd > 1:
            img_pd_bsn = pixel_shuffle_up_sampling(pd_img_denoised, f=pd, pad=self.pd_pad)
            img_pd_bsn = img_pd_bsn[:, :, :h, :w]
        else:
            p = self.pd_pad
            img_pd_bsn = pd_img_denoised[:, :, p:-p, p:-p]
        return img_pd_bsn

    def forward(self, img, pd=None):
        return self.forward_pd(img, pd=pd)

    def denoise(self, x):
        b, c, h, w = x.shape
        if h % self.pd_b != 0:
            x = F.pad(x, (0, 0, 0, self.pd_b - h % self.pd_b), mode='constant', value=0)
        if w % self.pd_b != 0:
            x = F.pad(x, (0, self.pd_b - w % self.pd_b, 0, 0), mode='constant', value=0)
        hp, wp = x.shape[-2:]
        if hp % self.pd_b != 0 or wp % self.pd_b != 0:
            raise RuntimeError('AP-BSN denoise padding failed for pd_b=%s, got shape %s' % (self.pd_b, tuple(x.shape)))
        img_pd_bsn = self.forward_pd(img=x, pd=self.pd_b)
        return img_pd_bsn[:, :, :h, :w]

class APBSNForwardMixin(object):

    def __init__(self, *args, forward_mode='auto', train_forward_mode='forward', eval_forward_mode='denoise', **kwargs):
        self.forward_mode = forward_mode
        self.train_forward_mode = train_forward_mode
        self.eval_forward_mode = eval_forward_mode
        super().__init__(*args, **kwargs)

    def _current_forward_mode(self):
        if self.forward_mode == 'auto':
            return self.train_forward_mode if self.training else self.eval_forward_mode
        return self.forward_mode

    def forward(self, img, pd=None):
        mode = self._current_forward_mode()
        if pd is not None or mode in ('forward', 'pd', 'train'):
            out = self.forward_pd(img, pd=pd)
        elif mode in ('denoise', 'inference', 'test'):
            out = self.denoise(img)
        else:
            raise ValueError('Unsupported AP-BSN forward mode: %s' % mode)
        return out

    def load_state_dict(self, state_dict, strict=True):
        if isinstance(state_dict, dict) and 'denoiser' in state_dict and isinstance(state_dict['denoiser'], dict):
            state_dict = state_dict['denoiser']
        if isinstance(state_dict, dict) and 'model_weight' in state_dict and isinstance(state_dict['model_weight'], dict):
            model_weight = state_dict['model_weight']
            if 'denoiser' in model_weight:
                state_dict = model_weight['denoiser']
        return super().load_state_dict(state_dict, strict=strict)

class APBSNFrozenMixin(object):

    def __init__(self, *args, frozen=True, force_eval=True, **kwargs):
        self.frozen = frozen
        self.force_eval = force_eval
        super().__init__(*args, **kwargs)
        if self.frozen:
            for param in self.parameters():
                param.requires_grad_(False)
            super().train(False)

    def train(self, mode=True):
        if self.force_eval:
            return super().train(False)
        return super().train(mode)

@BACKBONE.register_module()
class APBSNBackbone(APBSNForwardMixin, APBSNCore):

    def __init__(self, *args, blindspot=True, forward_mode='auto', train_forward_mode='forward', eval_forward_mode='denoise', **kwargs):
        super().__init__(*args, forward_mode=forward_mode, train_forward_mode=train_forward_mode, eval_forward_mode=eval_forward_mode, **kwargs)
        self.blindspot = blindspot
        for module in self.modules():
            if isinstance(module, CentralMaskedConv2d):
                module.blindspot = blindspot

@BACKBONE.register_module()
class APBSNTeacher(APBSNFrozenMixin, APBSNForwardMixin, APBSNCore):

    def __init__(self, *args, forward_mode='denoise', train_forward_mode='denoise', eval_forward_mode='denoise', frozen=True, force_eval=True, **kwargs):
        super().__init__(*args, forward_mode=forward_mode, train_forward_mode=train_forward_mode, eval_forward_mode=eval_forward_mode, frozen=frozen, force_eval=force_eval, **kwargs)
