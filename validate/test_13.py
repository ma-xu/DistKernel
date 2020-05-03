import torch
import torch.nn as nn
from collections import OrderedDict
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F
import math
# import models
# from models.ResNet.resnet_dist9 import DPConv

### Fuse BN and Conv:

# New conv_weights: W = W_{bn}*W_{conv}
# New conv_bias:    b = W_{bn}*b_{conv}+b_{bn}

# data = torch.rand(2,3,8,8)
# conv = nn.Conv2d(3,6,3)
# fuseConv = nn.Conv2d(3,6,3)
# bn = nn.BatchNorm2d(6)
# W_conv = conv.weight.clone()
# W_bn  =bn.weight.clone()
# b_conv = conv.bias.clone()
# b_bn = bn.bias.clone()
# W_fuse = W_bn.view(6,1,1,1)*W_conv
# b_fuse = W_bn*b_conv+b_bn
#
# fuseConv.weight = nn.Parameter(W_fuse)
# fuseConv.bias = nn.Parameter(b_fuse)
# out1 = bn(conv(data))
# out2 = fuseConv(data)
# print(out1)
# print(out2)

def fuse(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    beta = bn.weight
    gamma = bn.bias

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean)/var_sqrt * beta + gamma
    fused_conv = nn.Conv2d(conv.in_channels,
                         conv.out_channels,
                         conv.kernel_size,
                         conv.stride,
                         conv.padding,
                         bias=True)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv
x = torch.randn([2, 3, 6, 6])
conv = nn.Conv2d(3,6,3)
bn = nn.BatchNorm2d(6)
out1 = bn(conv(x))
fusedconv = fuse(conv,bn)
out2 = fusedconv(x)
print(out1)
print(out2)
