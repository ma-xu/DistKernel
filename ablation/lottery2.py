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

from WeightConvert import *
from utils.visualize_kernel import Distribution
import matplotlib.pyplot as plt









model_path = "/Users/melody/Desktop/Experiments/ParaDise/new1_resnet18_FP32_model_best.pth.tar"
check_point = torch.load(model_path,map_location=torch.device('cpu') )

resnet18 = models.__dict__['old_resnet18']()
from collections import OrderedDict
new_check = OrderedDict()
for k, v in check_point['state_dict'].items():
    name = k[7:]  # remove `module.`
    if '.conv1' in name or '.conv2' in name:
        # print(name)
        if '.group_conv.weight' in name:
            v=convert_group_to_standard(v,group=64)
        if '.ac_convbn.ver_conv.weight' in name:
            v=convert_vertical_to_standard(v)
        if '.ac_convbn.hor_conv.weight' in name:
            v=convert_horizontal_to_standard(v)
        # if '.num_batches_tracked' in name:
            # print(v)
    new_check[name] = v

save_check = OrderedDict()
for k, v in new_check.items():
    if '.conv1' in k or '.conv2' in k:
        if '.conv1' in k:
            index ='1'
        if '.conv2' in k:
            index = '2'
        prefix_key = k[0:9]   #layer4.1.

        if k.endswith('conv.weight'):
            new_key = prefix_key+'conv'+index+'.weight'
            if new_key in save_check:
                save_check[new_key] = torch.cat([save_check[new_key],v.unsqueeze(dim=-1)],dim=-1)
            else:
                save_check[new_key] = v.unsqueeze(dim=-1)

        if k.endswith('bn.weight'):
            new_key = prefix_key + 'bn' + index + '.weight'
            if new_key in save_check:
                save_check[new_key] = torch.cat([save_check[new_key],v.unsqueeze(dim=-1)],dim=-1)
            else:
                save_check[new_key] = v.unsqueeze(dim=-1)

        if k.endswith('bn.bias'):
            new_key = prefix_key + 'bn' + index + '.bias'
            if new_key in save_check:
                save_check[new_key] = torch.cat([save_check[new_key],v.unsqueeze(dim=-1)],dim=-1)
                # print("cat shape {}".format((save_check[new_key]).shape))
            else:
                save_check[new_key] = v.unsqueeze(dim=-1)
                # print("new shape {}".format((save_check[new_key]).shape))

        if k.endswith('bn.running_mean'):
            new_key = prefix_key + 'bn' + index + '.running_mean'
            save_check[new_key] = torch.zeros(v.shape)

        if k.endswith('bn.running_var'):
            new_key = prefix_key + 'bn' + index + '.running_var'
            save_check[new_key] = torch.ones(v.shape)

        if k.endswith('bn.num_batches_tracked'):
            new_key = prefix_key + 'bn' + index + '.num_batches_tracked'
            save_check[new_key] = torch.tensor(0)

    else:
        save_check[k] = v



for k, v in save_check.items():
    # print(k)
    if k.endswith('.conv1.weight') or k.endswith('.conv2.weight'):
         # save_check[k] = v.max(dim=-1,keepdim=False)
         value, index =v.max(dim=-1,keepdim=False)
         save_check[k] = nn.init.kaiming_normal_(value, mode='fan_out', nonlinearity='relu')
    if k.endswith('.bn1.weight') or k.endswith('.bn2.weight') \
            or k.endswith('.bn1.bias') or k.endswith('.bn2.bias'):
        # save_check[k] = v.max(dim=-1,keepdim=False)
        value, index = v.max(dim=-1, keepdim=False)
        save_check[k] = value

torch.save(save_check, 'normal_lottery_resnet18.pth.tar')
resnet18.load_state_dict(save_check)


