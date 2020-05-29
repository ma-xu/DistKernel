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
import torchvision

from WeightConvert import *
from utils.visualize_kernel import Distribution
import matplotlib.pyplot as plt




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
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


path = "/Users/melody/Desktop/Experiments/ParaDise/new1_resnet18_FP32_model_best.pth.tar"
check_point = torch.load(path,map_location=torch.device('cpu') )
check_point = check_point['state_dict']


minv=0
maxv=0

for k, v in check_point.items():
    print(k)
    if k.endswith("ori_conv.weight"):
        minv = min(v.float().min(),minv)
        maxv = max(v.float().max(),maxv)

for k, v in check_point.items():
    if k.endswith("ori_conv.weight"):
        hist= torch.histc(v.float(), bins=100,min=minv,max=maxv)

        plt.bar(range(len(hist)), hist,color='0.4')
        # plt.savefig('1.jpg')
        # plt.bar(torch.range(start=minv,end=maxv,step=(maxv-minv)/100.0), hist)
        # plt.bar(torch.range(start=minv,end=maxv,step=(maxv-minv)/100), hist)
        plt.show()


