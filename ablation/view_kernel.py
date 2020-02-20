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
from models.ResNet.resnet_dist5 import DPConv

model = models.__dict__['dist5_resnet18']()
path = "/Users/melody/Downloads/epoch96checkpoint.pth.tar"
check_point = torch.load(path,map_location='cpu')
new_check_point = OrderedDict()

for k, v in check_point['state_dict'].items():
    # name = k[7:]  # remove `module.`
    name = k[7:]  # remove `module.1.`
    new_check_point[name] = v
model.load_state_dict(new_check_point)


X = np.linspace(-1,1,60)
Y = np.linspace(-1,1,60)
X1 = torch.tensor(X).reshape(60, 1).expand((60, 60))
Y1 = torch.tensor(Y).reshape(1, 60).expand((60,60))
mask = torch.sqrt(X1.pow(2) + Y1.pow(2))
mask = mask.unsqueeze(dim=-1).unsqueeze(dim=-1)

for m in model.modules():
    if isinstance(m,DPConv):
        m.mask=mask
        param = m._init_distribution()
        param = param.mean(dim=0).mean(dim=0)
        param = param.detach().numpy()
        # print(param.shape)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, param, rstride=3, cstride=3, linewidth=1, antialiased=False)
        plt.show()
        print(param.shape)
        plt.close(fig)




N = 60
X = np.linspace(-2,2,N)
Y = np.linspace(-2,2,N)
X,Y = np.meshgrid(X,Y)
mu = np.array([0.,0.])
Sigma = np.array([[1.,-0.5],[-0.5,1.5]])
pos = np.empty(X.shape+(2,))
pos[:,:,0]= X
pos[:,:,1] = Y

p2 = Distribution(mu,Sigma)
Z = p2.tow_d_gaussian(pos)

fig =plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,Z,rstride=3,cstride=3,linewidth=1,antialiased =False)
# cset = ax.contour(X,Y,Z,zdir='z',offset=-0.15)

ax.set_zlim(0,0.2)
ax.set_zticks(np.linspace(0,0.2,5))
ax.view_init(27,-21)
plt.show()
