from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.fc1 = nn.Linear(784, 784)
        self.fc2 = nn.Linear(784, 1568)
        self.fc3 = nn.Linear(1568, 784)
        self.fc4 = nn.Linear(784, 128)
        self.dropout = nn.Dropout2d(0.5)
        self.fc5 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc5(x)
        output = F.log_softmax(x, dim=1)
        return output


def demo():
    net = Model1()
    x = torch.rand([1,1,28,28])
    y = net(x)
    print(y)

# demo()
