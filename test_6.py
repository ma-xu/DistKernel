import torch
import torch.nn as nn
from collections import OrderedDict
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F
import math

k=3;

def mask(k):
    y= torch.arange(0,k)-k//2+0.0
    y1 = y.reshape(k, 1).expand((k, k))
    y2 = y.reshape(1,k).expand((k, k))
    out = torch.sqrt(y1.pow(2)+y2.pow(2))
    return out
print(mask(k))
print(mask(k).shape)
