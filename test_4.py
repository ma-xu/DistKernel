import torch
import torch.nn as nn
from collections import OrderedDict
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F
import math

A =torch.rand(1,10,10)
B = torch.cat((2*A,A),dim=0)
print(B.shape)
print(A.var())
print(B.var())
