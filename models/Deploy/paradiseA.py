import torch.nn as nn
import torch
from acblock import *

class ParaDiseAConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False,deploy=False):
        super(ParaDiseAConv, self).__init__()
        self.deploy = deploy
        if deploy:
            # Similarly, BN should be fused in conv during inference for efficiency.
            self.paradise_a_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        else:
            self.ori_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
            self.new_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
            self.ac_convbn = ACBlock(in_channels,out_channels,kernel_size,stride,padding,deploy=deploy)
            self.group_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=64, bias=bias)

            self.ori_bn =  nn.BatchNorm2d(out_channels)
            self.new_bn = nn.BatchNorm2d(out_channels)
            self.group_bn = nn.BatchNorm2d(out_channels)



    def forward(self, input):
        if self.deploy:
            return self.paradise_a_conv(input)
        else:
            return self.ori_bn(self.ori_conv(input))+self.new_bn(self.new_conv(input))\
               +self.ac_convbn(input)+self.group_bn(self.group_conv(input))
