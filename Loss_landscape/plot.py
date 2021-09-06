import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from numpy import genfromtxt
import torch
import torch.nn.functional as F

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
data = genfromtxt('old_resnet18out.txt', delimiter=',')
where_are_NaNs = np.isnan(data)
data[where_are_NaNs] = 10000000
data = np.clip(data, a_min=-99999, a_max=100000000)


X = np.arange(-0.5, 0.6, 0.1)
Y = np.arange(-0.5, 0.6, 0.1)
X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)
log_base = 2
Z = -np.log(data[:,3])/np.log(log_base)
Z = Z.reshape(11,11)
Z = torch.from_numpy(Z).unsqueeze(dim=0).unsqueeze(dim=0)
ZZ = F.interpolate(Z, scale_factor=10)
print(Z.size)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)

# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

ax.get_xaxis().set_visible(True)
ax.axes.get_yaxis().set_visible(True)
# plt.axis('off')
plt.show()
fig.savefig("3d_ball.pdf", bbox_inches='tight', pad_inches=-0.5, transparent=True)
