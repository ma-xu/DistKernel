import torch
import torch.nn as nn
from collections import OrderedDict
from  torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.parameter import Parameter
import numpy as np


A = torch.tensor([1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,])
print(A)
print("A.var {}".format(A.std()))
B = torch.tensor([1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,])
print("B.var {}".format(B.std()))
C = torch.tensor([1.,2.])
print("C.var {}".format(C.std()))

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



