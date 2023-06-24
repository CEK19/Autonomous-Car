import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import copy
import sys

# Define the input data points
x = []
y = []
z = []
t1 = []
t2 = []

file = open("D:/NDT/PY_ws/DCLV/src/log.txt", "r")
data = file.read()
data = data.split("\n")
file.close()
for each in data:
    tmp = each.split(" ")
    if len(tmp) == 1:
        continue
    x.append(float(tmp[0]))
    y.append(float(tmp[1]))
    z.append(float(tmp[3]))
    t1.append(float(tmp[2]))
    t2.append(float(tmp[3]))

# x = np.array(x)
# y = np.array(y)
# z = np.array(z)



# Create a function for bicubic interpolation net5.0-windows
f = interp2d(x, y, z, kind='cubic')

# Create a grid for the output
xi = np.arange(0.01, 1.01, 0.01)
yi = np.arange(0.01, 2.01, 0.01)
X, Y = np.meshgrid(xi, yi)

# Compute the interpolated values
Z = f(xi, yi)
max_index = np.unravel_index(np.argmax(Z), Z.shape)
# print(X[max_index],Y[max_index])
# Z = Z*0.5+0.5

# Z = np.clip(Z,0,2)

# Plot the interpolated surface
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z, cmap='viridis')
# ax.set_xlabel('Alpha')
# ax.set_ylabel('Gamma')
# ax.set_zlabel('avg')
# plt.show()

copyZ = copy.deepcopy(z)
copyZ.sort()
for _ in range(10):
    idx = z.index(max(copyZ))
    print(x[idx],y[idx],z[idx] , " --- ",t1[idx],t2[idx])
    copyZ[copyZ.index(max(copyZ))] = min(copyZ)
# plt.plot(z)
# plt.show()