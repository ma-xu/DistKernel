# select the max value of the ParaDise
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
from utils.visualize_kernel import Distribution
import matplotlib.pyplot as plt

model_path = "/Users/melody/Desktop/Experiments/ParaDise/new1_resnet18_FP32_model_best.pth.tar"
check_point = torch.load(model_path,map_location=torch.device('cpu') )

resnet18 = models.__dict__['old_resnet18']()
new1_resnet18 = models.__dict__['new1_resnet18']()
from collections import OrderedDict
new_check = OrderedDict()
for k, v in check_point['state_dict'].items():
    name = k[7:]  # remove `module.`
    new_check[name] = v
new1_resnet18.load_state_dict(new_check)
print(new1_resnet18)
