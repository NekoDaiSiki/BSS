import torch
import torch.nn as nn
import torch.nn.init as init

import torch.nn.functional as F
from src.utils.register import Registry
from src.utils.builder import HEAD

from src.model.basic.basic_model import CentralMaskedConv2d

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5

@HEAD.register_module()
class EnhBlock(nn.Module):
    def __init__(self, c_in=3, c_mid=128, c_out=3):
        super(EnhBlock, self).__init__()
        self.layers = nn.Sequential(
            DenseBlock(c_in, c_mid),
            nn.Conv2d(c_mid, c_mid, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(c_mid, c_mid, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(c_mid, c_mid, kernel_size=1, stride=1, padding=0, bias=True),
            DenseBlock(c_mid, c_out)
        )

    def forward(self, x):
        return x + self.layers(x) * 0.2
    

@HEAD.register_module()
class CentralMaskBlock(nn.Module):
    def __init__(self,c_in=3,c_mid=32,c_out=3):
        super().__init__()
        self.cm_conv = CentralMaskedConv2d(c_in,c_mid,kernel_size=3,padding=1)
        self.conv1 = nn.Conv2d(c_mid,c_mid,kernel_size=1)
        self.conv2 = nn.Conv2d(c_mid,c_mid,kernel_size=1)
        self.conv3 = nn.Conv2d(c_mid,c_out,kernel_size=1)
        self.lkrelu = nn.LeakyReLU()
    
    def forward(self,x):
        x = self.cm_conv(x)
        res = self.conv1(x)
        res = self.lkrelu(res)
        res = self.conv2(res)
        x = x  + res
        x = self.conv3(x)
        return x