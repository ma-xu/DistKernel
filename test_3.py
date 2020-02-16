import torch
import torch.nn as nn
from collections import OrderedDict
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F
import math
#
# def get_location_mask(x):
#     c_in,c_out,w,h = x.shape
#     mask = (x[0, 0, :, :] != -999).nonzero()
#     mask = mask.reshape(w, h, 2)
#     mask  =mask - w//2 # assume square, lazy.
#     mask = mask.expand(c_in*c_out,w,h,2).permute(1,2,0,3)
#     return mask
# conv1 = torch.nn.Conv2d(3,3,3,padding=1,stride=1)
# param = conv1.weight
# mask  = get_location_mask(param)
#
# std = math.sqrt(2)/math.sqrt(8*3*3)
# m = MultivariateNormal(loc=torch.zeros(2),scale_tril=std*torch.ones(9,2).diag_embed())
# # print(m.log_prob(torch.rand(2,4,2,2)).exp())
# print(m.log_prob(torch.tensor([0,0])).exp())
#
# exit()






def get_location_mask(x):
    c_in,c_out,w,h = x.shape
    mask = (x[0, 0, :, :] != -999).nonzero()
    mask = mask.reshape(w, h, 2)
    mask  =mask - w//2 # assume square, lazy.
    # mask = mask.expand(c_in*c_out,w,h,2).permute(1,2,0,3)
    return mask.unsqueeze(dim=2)
conv1 = torch.nn.Conv2d(3,64,3,padding=1,stride=1)
param = conv1.weight
mask  = get_location_mask(param)
normal_loc = torch.zeros(2)
std =math.sqrt(2)/math.sqrt(param.shape[0]*param.shape[2]*param.shape[3])
scale_tril = torch.ones(param.shape[0]*param.shape[1],2)
normal_scal = 1*scale_tril

m = MultivariateNormal(loc=normal_loc,scale_tril=(normal_scal).diag_embed())
y = m.log_prob(mask).exp()
y = y.permute(2,0,1).reshape(param.shape)

y = -(y-F.adaptive_avg_pool2d(y,1))
# print(y)
y = math.sqrt(std/y.var())*y
print(y.var())
print(std)
print(y.shape)




# normal = Normal(0,1)
# x=torch.rand(10000)
# y = normal.log_prob(x).exp()
# z = normal.sample([10000])
# print(y)
# print(y.var())
# print(z.var())


#
# def get_location_mask(x):
#     mask = (x[0, 0, :, :] != -999).nonzero()
#     mask = mask.reshape(x.shape[2], x.shape[3], 2)
#     offset = torch.tensor([mask])
#     return mask
#
# conv1 = torch.nn.Conv2d(3,3,3,padding=1,stride=1)
# param = conv1.weight
# mask  = get_location_mask(param)
# D = MultivariateNormal(loc=torch.zeros(2),scale_tril=torch.ones(4,2).diag_embed())
# out = D.sample([10,10])
# density = D.log_prob(out).exp()
# print(density)
# print(out.shape)
# print(torch.mean(out))
# print(torch.var(out))
# print(D.stddev)
#
#
#
# # tensor  = torch.rand((10000,10000))
# # tensor = tensor.normal_(0, 1)
# # print(torch.mean(tensor))
# # print(torch.var(tensor))
#
#
#
#
# # A = torch.tensor([1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,])
# # print(A)
# # print("A.var {}".format(A.std()))
# # B = torch.tensor([1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,])
# # print("B.var {}".format(B.std()))
# # C = torch.tensor([1.,2.])
# # print("C.var {}".format(C.std()))

# D = torch.tensor([1.,2.,3.,4.])

#
# kernel_size = 3;
# normal_loc = torch.zeros(2)
# normal_scal =torch.ones(2)
#
# m = MultivariateNormal(loc=normal_loc,scale_tril=(normal_scal).diag_embed())
# print(m.sample((3,3)))
#
#
# #
# #
# #
# data = torch.rand(1,3,5,5)
# conv1 = torch.nn.Conv2d(3,3,3,padding=1,stride=1)
# nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='relu')
# # print(conv1(data))
#
# y = torch.rand((4,4))
# print(y.shape)
# y = y.normal_(0.0,10.0)
# print(y)
#
# print(y.std())
#
#
#
#
#



