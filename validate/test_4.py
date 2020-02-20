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



# A =torch.rand(1,10,10)
# B = torch.cat((2*A,A),dim=0)
# print(B.shape)
# print(A.var())
# print(B.var())

model = models.__dict__['dist5_resnet18']()
path = "/Users/melody/Downloads/epoch96checkpoint.pth.tar"
check_point = torch.load(path,map_location='cpu')
new_check_point = OrderedDict()

for k, v in check_point['state_dict'].items():
    # name = k[7:]  # remove `module.`
    name = k[7:]  # remove `module.1.`
    new_check_point[name] = v
model.load_state_dict(new_check_point)

y = model(torch.randn(2, 3, 224,224))
print(y.size())

# print(check_point['best_prec1'])
# exit()

for k, v in check_point['state_dict'].items():
    if "distribution_std" in k:
        # print("{} : {}".format(k,v))
        # print(v.data)
        print(v.mean())
    # if "distribution_var" in k:
    #     print("{} : {}".format(k,v))
    # if "normal_loc" in k:
    #     print("{} : {}".format(k,v))

