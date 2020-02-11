import torch
from collections import OrderedDict

data = torch.rand(1,3,5,5)
conv1 = torch.nn.Conv2d(3,3,3,padding=1,stride=1)
param1 = conv1.state_dict()
conv2 = torch.nn.Conv2d(3,3,3,padding=1,stride=1)
param2 = conv2.state_dict()
conv3 = torch.nn.Conv2d(3,3,3,padding=1,stride=1)

new_check_point = OrderedDict()
for k, v in param1.items():
    new_check_point[k] = v
    new_check_point[k] = new_check_point[k] +param2[k]
# print(param1)
# print(param2)
# print(new_check_point)
conv3.load_state_dict(new_check_point)
y1 = conv1(data)+conv2(data)
y2= conv3(data)
print(y1)
print(y2)
print(y1==y2)












