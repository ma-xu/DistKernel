import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt

# Make data.
filename="new1_resnet18loss"
data = genfromtxt(f'{filename}.txt', delimiter=',')
data = data[:,2]
n = int(np.sqrt(data.size))
data = data.reshape(n,n)

plt.imshow(data, cmap='viridis')
plt.colorbar()
plt.show()


data1 = data[:,n-1]
print(f"direction 1 is \n{filename}:")
for i in data1:
    print(i)


data2 = data[n-1,:]
print(f"direction 2 is \n{filename}:")
for i in data2:
    print(i)
