import torch
import sys
sys.path.append("../")
import torch.nn as nn
from collections import OrderedDict
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F
import models as model
import cv2
from models.ResNet.resnet_new1 import *

img_path = "/Users/melody/Downloads/cat.jpeg"
path = "/Users/melody/Desktop/Experiments/ParaDise/new1_resnet18_FP32_model_best.pth.tar"
check_point = torch.load(path,map_location=torch.device('cpu') )
check_point = check_point['state_dict']
new_check = OrderedDict()
for k, v in check_point.items():
    name = k[7:]  # remove `module.`
    new_check[name] = v
    # print(name)

selected_layer = 'layer4.1.conv2.'

ori_weight      = new_check['layer4.1.conv2.'+'ori_conv.weight']
new_weight      = new_check['layer4.1.conv2.'+'new_conv.weight']
group_weight    = new_check['layer4.1.conv2.'+'group_conv.weight']
ver_weight      = new_check['layer4.1.conv2.'+'ac_convbn.ver_conv.weight']
hor_weight      = new_check['layer4.1.conv2.'+'ac_convbn.hor_conv.weight']
hor_weight = hor_weight.transpose(2,3)
ac_weight = torch.cat([ver_weight,hor_weight],dim=-1)
min_value = min(ori_weight.min(),new_weight.min(),group_weight.min(),ac_weight.min())
max_value = max(ori_weight.max(),new_weight.max(),group_weight.max(),ac_weight.max())
print(ver_weight.shape)
print(hor_weight.shape)

ori_hist = torch.histc(ori_weight, bins=20, min=min_value, max=max_value).unsqueeze(dim=0)
new_hist = torch.histc(new_weight, bins=20, min=min_value, max=max_value).unsqueeze(dim=0)
group_hist = torch.histc(group_weight, bins=20, min=min_value, max=max_value).unsqueeze(dim=0)
ac_hist = torch.histc(ac_weight, bins=20, min=min_value, max=max_value).unsqueeze(dim=0)
all_hist = torch.cat([ori_hist,new_hist,group_hist,ac_hist],dim=0)
print(all_hist)
