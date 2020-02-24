import torch
from collections import OrderedDict
import torchvision.models

data = torch.rand(1,3,5,5)
conv1 = torch.nn.Conv2d(3,6,3,padding=1,stride=1)
param1 = conv1.state_dict()
conv2 = torch.nn.Conv2d(3,3,3,padding=1,stride=1)


param2 = OrderedDict()
param2['weight'] = param1['weight'].view(2,3,3,3,3)
for k, v in param1.items():
    shape = v.shape
    v = v.view(2,shape[0],shape[1:])
    param2[k] = v
    print(v.shape)
    param2[k] = param2[k]
# print(param1)
# print(param2)
# print(new_check_point)










