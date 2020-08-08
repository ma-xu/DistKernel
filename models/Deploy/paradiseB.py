import torch.nn as nn
import torch
from acblock import *
from WeightConvert import *

class ParaDiseBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=False,deploy=False):
        super(ParaDiseBConv, self).__init__()
        self.deploy = deploy

        self.ori_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.new_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.ac_convbn = ACBlock(in_channels,out_channels,kernel_size,stride,padding,deploy=deploy)
        self.group_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=64, bias=bias)

        self.ori_bn =  nn.BatchNorm2d(out_channels)
        self.new_bn = nn.BatchNorm2d(out_channels)
        self.group_bn = nn.BatchNorm2d(out_channels)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, 4),
            nn.Softmax(dim=1)
        )

        if deploy:
            # Similarly, BN should be fused in conv during inference for efficiency.
            self.paradise_b_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
            self.register_buffer("ori_weight", fuse(self.ori_conv,self.ori_bn).weight)
            self.register_buffer("new_weight", fuse(self.new_conv, self.new_bn).weight)
            self.register_buffer("ac_weight", self.ac_convbn.ac_conv.weight)
            self.register_buffer("group_weight", convert_group_to_standard(fuse(self.group_conv, self.group_bn)).weight)



    def forward(self, input):
        gap = input.mean(dim=-1).mean(dim=-1)
        weights = self.fc(gap).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        if self.deploy:
            paradise_b = weights[:,0,:,:,:]*self.ori_weight + weights[:,1,:,:,:]*self.new_weight\
                         + weights[:,2,:,:,:]*self.ac_weight + weights[:,3,:,:,:]*self.group_weight
            self.paradise_b_conv.weight = nn.Parameter(paradise_b)
            return self.paradise_b_conv(input)
        else:
            ori_out = self.ori_bn(self.ori_conv(input)).unsqueeze(dim=1)
            new_out = self.new_bn(self.new_conv(input)).unsqueeze(dim=1)
            ac_out = self.ac_convbn(input).unsqueeze(dim=1)
            group_out = self.group_bn(self.group_conv(input)).unsqueeze(dim=1)
            all_out = torch.cat([ori_out, new_out, ac_out, group_out], dim=1)
            out = weights * all_out
            out = out.sum(dim=1, keepdim=False)
            return out
