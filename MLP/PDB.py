from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

class Ass(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Ass, self).__init__()
        self.branch1 = nn.Linear(inchannel, outchannel)
        self.branch2 = nn.Linear(inchannel, outchannel)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // 16),
            nn.ReLU(inplace=True),
            nn.Linear(inchannel // 16, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, input):
        out1 = self.branch1(input).unsqueeze(dim=1)
        out2 = self.branch2(input).unsqueeze(dim=1)

        all_out = torch.cat([out1, out2], dim=1)

        weights = self.fc(input).unsqueeze(dim=-1)

        out = weights * all_out
        out = out.sum(dim=1, keepdim=False)
        return out


class PDB(nn.Module):
    def __init__(self):
        super(PDB, self).__init__()
        self.fc1 = Ass(784, 64)
        self.fc2 = Ass(64, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def demo():
    net = PDB()
    x = torch.rand([1,1,28,28])
    y = net(x)
    print(y)

# demo()
