import torch
import torch.nn as nn
from collections import OrderedDict
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F
import math

# A =torch.rand(1,10,10)
# B = torch.cat((2*A,A),dim=0)
# print(B.shape)
# print(A.var())
# print(B.var())


path = "/Users/melody/Downloads/checkpoint_dist4.pth.tar"
check_point = torch.load(path,map_location='cpu')

print(check_point['best_prec1'])
exit()

for k, v in check_point['state_dict'].items():
    if "distribution_std" in k:
        # print("{} : {}".format(k,v))
        print(v.data)
    # if "distribution_var" in k:
    #     print("{} : {}".format(k,v))
    # if "normal_loc" in k:
    #     print("{} : {}".format(k,v))

