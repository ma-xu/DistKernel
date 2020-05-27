import torch
import torch.nn as nn
import torch.distributions.normal
import torchvision.models


input =torch.rand([2,3,5,5])

conv1 = nn.Conv2d(3,4,3)

bn1 = nn.BatchNorm2d(4)
nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='relu')
nn.init.normal_(bn1.weight)
out = bn1(conv1(input)).mean(dim=-1).mean(dim=-1)
print(out)
print()
# use  mean/std in BN layers

