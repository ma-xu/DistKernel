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

# Explore the method that can describe a tensor.
# a=torch.tensor([1.,2.])
# b=2*a
# print(a.softmax(dim=0))
# print(b.softmax(dim=0))
# exit(0)


data =  torch.rand(3,1,8,8)
conv = nn.Conv2d(1,1,3,bias=False)
pooling = nn.AdaptiveAvgPool2d(1)
out1 = pooling(data)
out2 = pooling(conv(data))
# weight =
print(out1)
print(out2)




exit(0)



data = torch.rand([1,16,8,8])
conv1 = nn.Conv2d(16,8,3,bias=False)
conv2 = nn.Conv2d(16,8,3,bias=False)
gap = nn.AdaptiveAvgPool2d(1)
weight1 = conv1.weight
weight2 = conv2.weight
out1 = conv1(data)
gap_out1 = gap(out1).unsqueeze(dim=1)
out2 = conv2(data)
gap_out2 = gap(out2).unsqueeze(dim=1)
gap_allout = torch.cat([gap_out1,gap_out2],dim=1)
softout=gap_allout.softmax(dim=1).squeeze(dim=-1).squeeze(dim=-1)
# print(softout)
print(softout.shape)
print(softout)
#
# wei1 = weight1.mean(dim=-1).mean(dim=-1).mean(dim=-1)
# print(wei1.shape)


# a = torch.rand([128,64,1,1])
#
# # SVD
# mm = torch.svd(a)
# print(mm.S.shape)
# torch.norm()
