import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import torch
import torch.nn.functional as F
from matplotlib import rc

plt.rcParams["font.family"] = "Times New Roman"

# Make data.
filename="old_resnet18loss"
data = genfromtxt(f'{filename}.txt', delimiter=',')
where_are_NaNs = np.isnan(data)
data[where_are_NaNs] = 10000000
data = np.clip(data, a_min=-99999, a_max=10000000)
Z = data[:, 2]
# Z = (Z-np.mean(Z))/(np.std(Z)+1e-8)
assert Z.size == int(np.sqrt(Z.size)) * int(np.sqrt(Z.size))
Z = Z.reshape(int(np.sqrt(Z.size)), int(np.sqrt(Z.size)) )
Z = torch.from_numpy(Z).unsqueeze(dim=0).unsqueeze(dim=0)
Z = F.interpolate(Z, scale_factor=10, mode='bicubic')
Z = Z.squeeze(dim=0).squeeze(dim=0).numpy()
print(Z.size)
X = np.arange(-0.3, 0.33, 0.003)
Y = np.arange(-0.3, 0.33, 0.003)
X, Y = np.meshgrid(X, Y)


fig, ax = plt.subplots(figsize=(4,3))
line_lv = [1.5,2.0,2.5,3.0,4,5,6,7]
cf_lv = [0,1.5,2,2.5,3,4,5,6,7,8]
# CS = ax.contourf(X, Y, Z)
plt.contourf(X, Y, Z, levels=cf_lv, alpha=.9, cmap='autumn')
# CS = ax.contour(X, Y, Z)
CS = plt.contour(X, Y, Z, levels=line_lv, colors='black', linewidths=.5)

# ax.set_xlabel('$direction 1$', family='fantasy'); ax.set_ylabel('$direction 2$')
#设置x,y轴刻度
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.xticks(np.arange(-10,11,2))
# plt.yticks(np.arange(-10,11,2))
plt.clabel(CS, inline=True, fontsize=16)
# 避免图片显示不完全
plt.tight_layout()
plt.show()
fig.savefig(f"{filename}_contour.pdf", bbox_inches='tight', pad_inches=-0, transparent=True)
