import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from numpy import genfromtxt
import torch
import torch.nn.functional as F

fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(4,3))

# Make data.
filename="old_resnet18new_losses"
data = genfromtxt(f'{filename}.txt', delimiter=',')
where_are_NaNs = np.isnan(data)
data[where_are_NaNs] = 10000000
data = np.clip(data, a_min=-99999, a_max=1000)



# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)
log_base = 2
Z = data[:, 2]
# Z = (Z-np.mean(Z))/(np.std(Z)+1e-8)
assert Z.size == int(np.sqrt(Z.size)) * int(np.sqrt(Z.size))
Z = Z.reshape(int(np.sqrt(Z.size)), int(np.sqrt(Z.size)) )
Z = torch.from_numpy(Z).unsqueeze(dim=0).unsqueeze(dim=0)
Z = F.interpolate(Z, scale_factor=10, mode='bicubic')
Z = Z.squeeze(dim=0).squeeze(dim=0).numpy()
print(Z.size)
X = np.arange(0, int(np.sqrt(Z.size)), 1)
Y = np.arange(0, int(np.sqrt(Z.size)), 1)
X, Y = np.meshgrid(X, Y)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False, rcount=200, ccount=200)

# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

ax.get_xaxis().set_visible(True)
ax.axes.get_yaxis().set_visible(True)
plt.axis('off')
# fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
fig.savefig(f"{filename}3d_ball.pdf", bbox_inches='tight', pad_inches=-0.3, transparent=True)
