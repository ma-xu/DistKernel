import torch
import torch.nn as nn
from collections import OrderedDict
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F
import math

path = "/Users/melody/Downloads/epoch63checkpoint.pth.tar"
check_point = torch.load(path,map_location='cpu')

# print(check_point['best_prec1'])
# exit()

def _get_mask(k):
    # assume square and odd number, because lazy.
    y = torch.arange(0, k) - k // 2 + 0.0
    y1 = y.reshape(k, 1).expand((k, k))
    y2 = y.reshape(1, k).expand((k, k))
    mask = torch.sqrt(y1.pow(2) + y2.pow(2))
    return mask.unsqueeze(dim=-1).unsqueeze(dim=-1)

mask = _get_mask(3)

for k, v in check_point['state_dict'].items():
    if k.str.endswith(""):
        print(k)
        y = -(1.0 / (k.distribution_std * math.sqrt(2 * math.pi))) \
            * torch.exp(-((mask * k.distribution_zoom) ** 2) / (2 * (k.distribution_std ** 2)))
