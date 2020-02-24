import torch
import torch.nn as nn
from collections import OrderedDict
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F
import math
import models
from models.ResNet.resnet_dist9 import DPConv

model = models.__dict__['dist5_resnet18']()
path = "/Users/melody/Downloads/epoch49checkpoint.pth.tar"
check_point = torch.load(path,map_location='cpu')
new_check_point = OrderedDict()
model = models.dist9_resnet18()
for k, v in check_point['state_dict'].items():
    # name = k[7:]  # remove `module.`
    name = k[7:]  # remove `module.1.`
    new_check_point[name] = v
model.load_state_dict(new_check_point)

for m in model.modules():
    if isinstance(m, DPConv):
        # param = m._init_distribution()
        param = m.perturbation.weight
        c_out,c_out, k1,k2 = param.shape
        # print("True  Var: {}".format(param.var()))
        # print("Kaim  Var: {}".format(1/(c_out*k1*k2)))
        # print("True Mean: {}".format(param.mean()))
        # print("_____________________")
        print(param.var().detach().numpy())
