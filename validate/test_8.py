import torch
import torch.nn as nn
from collections import OrderedDict
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torchvision.models import *
from torchvision.models.mobilenet import ConvBNReLU


# path = "/Users/melody/Downloads/densenet121.pth"
# check_point = torch.load(path,map_location='cpu')
# for k, v in check_point.items():
#     if k.endswith("conv.2.weight"):
#         # print(v.mean().detach().numpy())
#         c_out,c_in,k1,k2 = v.shape
#         # print(2/(c_out*k1*k2))
#         print(v.shape)




# path = "/Users/melody/Downloads/vgg16.pth"
# check_point = torch.load(path,map_location='cpu')
# model = vgg16()
# model.load_state_dict(check_point)
# for m in model.modules():
#     if isinstance(m, nn.Conv2d):
#         # print(m.weight.mean().detach().numpy())
#         c_out,c_in,k1,k2 = m.weight.shape
#         print(2/(c_out*k1*k2))

#
# path = "/Users/melody/Downloads/mobilenet_v2.pth"
# check_point = torch.load(path,map_location='cpu')
# model = MobileNetV2()
# model.load_state_dict(check_point)
# for m in model.modules():
#     if isinstance(m, ConvBNReLU):
#         weight = m[0].weight
#         # print(weight.shape)
#         if weight.shape[3]==3:
#             # print(weight.mean().detach().numpy())
#             c_out, c_in, k1, k2 = m[0].weight.shape
#             print(2/(c_out*k1*k2))





path = "/Users/melody/Downloads/resnet18.pth"
check_point = torch.load(path,map_location='cpu')

# print(check_point['best_prec1'])
# exit()


for k, v in check_point.items():
    if k.endswith("conv2.weight") or k.endswith("conv1.weight"):
        # print(v.shape)
        out_channel, in_channel, k1,k2 = v.shape
        # print("Suppose var = {}".format(2/(out_channel*k1*k2)))
        # print("Real    var = {}".format(v.var()))
        # print("Real   mean = {}".format(v.mean()))
        # print("______________________")
        # print(v.mean().detach().numpy())
        print(2/(out_channel*k1*k2))


