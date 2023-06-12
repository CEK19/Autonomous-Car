import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import random

def isExis(x,y,valX,valY):
    for each in range(len(x)):
        if x[each] == valX and y[each] == valY:
            return True
    return False

def interpolate():
    x = []
    y = []
    z = []

    file = open("D:/NDT/PY_ws/DCLV/src/log.txt", "r")
    data = file.read()
    file.close()
    data = data.split("\n")
    if len(data) > 18:
        for each in data:
            tmp = each.split(" ")
            if len(tmp) == 1:
                continue
            x.append(float(tmp[0]))
            y.append(float(tmp[1]))
            z.append(float(tmp[3]) + float(tmp[2]))

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
    else:
        return round(np.random.rand()*0.8+0.1,2),round(np.random.rand()*1.8+0.1,2)

    # Create a function for bicubic interpolation net5.0-windows
    f = interp2d(x, y, z, kind='cubic')

    # Create a grid for the output
    xi = np.arange(0.01, 1.01, 0.01)
    yi = np.arange(0.01, 2.01, 0.01)
    X, Y = np.meshgrid(xi, yi)

    # Compute the interpolated values
    Z = f(xi, yi)
    max_index = np.unravel_index(np.argmax(Z), Z.shape)
    valX, valY = X[max_index],Y[max_index]

    while (isExis(x,y,valX, valY)):
        valX, valY = round(np.random.rand()*0.8+0.1,2),round(np.random.rand()*1.8+0.1,2)

    if (X[max_index] in x and Y[max_index] in y):
        return random.random()*0.8 + 0.1, random.random()*1.8 + 0.1
    return valX, valY
