import torch
import torch.nn as nn
from collections import OrderedDict
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F
import math

input = torch.rand(2,5,1,1)
# distribution = MultivariateNormal(loc=torch.zeros(2), scale_tril=(torch.ones(2)).diag_embed())
loc = torch.zeros(2,2)
scale = torch.rand(2,2)
distribution = Normal(loc=loc,scale=scale)
y1 = distribution.log_prob(input).exp()
print(y1)

def _norm_pdf(x, std):
    y = (1.0/(std*math.sqrt(2*math.pi)))*torch.exp(-(x**2)/(2*std**2))
    return y

y2 = _norm_pdf(input,scale)
print(y2)
print(y1.__str__()==y2.__str__())
