import torch
import torch.nn as nn
from collections import OrderedDict
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F
import math
# import models
# from models.ResNet.resnet_dist9 import DPConv

# Explore the method that can describe a tensor.


a = torch.rand([128,64,1,1])

# SVD
mm = torch.svd(a)
print(mm.S.shape)
torch.norm()
