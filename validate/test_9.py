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

def pdf(x):
    y = 1/math.sqrt(2*math.pi)*torch.exp(-1/(2*x*x))
    return y

a = torch.arange(-500,500)
a = a/100.0
y = pdf(a)
print(y.var())
y1 = pdf(torch.tensor(0.4)*5)
print(y1)
y2 = 5*pdf(torch.tensor(0.4))
print(y2)

exit()







y1 = torch.rand(300,1)*4
y1 = y1-y1.mean()
y2 = torch.rand(300,1)*2
y2 = y2-y2.mean()
y3 = torch.cat((y1,y2),dim=1)
print(y3.mean())
print("y1.var: {}".format(y1.var()))
print("y2.var: {}".format(y2.var()))
print("y3.var: {}".format(y3.var()))
print((y1.var()+y2.var())/2)

exit()

# y = torch.rand(20)
# yy = (y-y.mean())/y.std()
# print("y  std: {}".format(y.std()))
# print("yy std: {}".format(yy.std()))
# exit()




def cdf(x):
    y = (1.0 / math.sqrt(2 * math.pi)) \
        * torch.exp(-(x*x)/2)
    return y
input = torch.tensor([
    [math.sqrt(2),1,math.sqrt(2)],
    [1,0,1],
    [math.sqrt(2),1,math.sqrt(2)],
])
print(input.shape)
input = torch.rand((3,4,5))
out = cdf(input)
print(out)
print("mean: {}".format(out.mean()))
print("std: {}".format(out.std()))
print("test sqrt(2): {}".format(cdf(torch.tensor([math.sqrt(2)]))))
