# select the max value of the ParaDise
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F
import math
import models as models


#convert the group_conv weights to full_conv weights
def convert_group_to_standard(param,group=64):
    c_out,c_in,h,w = param.shape
    c_in = c_in*group

    param_full = torch.zeros((c_out, c_in, h, w))
    param_list = torch.chunk(param, group, dim=0)
    in_step = c_in // group
    out_step = c_out // group
    for i in range(0, group):
        param_full[i * out_step:(i + 1) * out_step, i * in_step:(i + 1) * in_step, :, :] = param_list[i]
    return param_full

#convert the vertical_conv weights to full_conv weights
def convert_vertical_to_standard(param,width=3):
    c_out, c_in, h, w = param.shape
    full_weight = torch.zeros([c_out, c_in, h, width])
    full_weight[:,:,:,(width-w)//2:((width-w)//2+w)] = param
    return full_weight


#convert the horizontal_conv weights to full_conv weights
def convert_horizontal_to_standard(param,height=3):
    c_out, c_in, h, w = param.shape
    full_weight = torch.zeros([c_out, c_in, height, w])
    full_weight[:,:,(height-h)//2:((height-h)//2+h),:] = param
    return full_weight




def demoVertical():
    vertical_conv = torch.nn.Conv2d(8, 16, (3,1), bias=False)
    weight = vertical_conv.weight
    # print(weight)
    full_weight = convert_vertical_to_standard(weight)
    print(full_weight)

def demoGroup():
    feature = torch.rand([2,8,7,7])
    group_conv = torch.nn.Conv2d(8,16,3,groups=4,bias=False)
    full_conv = torch.nn.Conv2d(8,16,3,bias=False)
    group_output = group_conv(feature)
    group_weight = group_conv.weight
    full_weight = convert_group_to_standard(group_weight,group=4)
    full_conv.weight = torch.nn.Parameter(full_weight)
    full_output = full_conv(feature)

    print(group_output)
    print(full_output)


def demoHorizontal():
    Horizontal_conv = torch.nn.Conv2d(8, 16, (1,3), bias=False)
    weight = Horizontal_conv.weight
    # print(weight)
    full_weight = convert_horizontal_to_standard(weight)
    print(full_weight)
