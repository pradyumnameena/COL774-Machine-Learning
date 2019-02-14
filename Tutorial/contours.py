# Adopted from: https://matplotlib.org/gallery/images_contours_and_fields/contourf_demo.html#sphx-glr-gallery-images-contours-and-fields-contourf-demo-py

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import math


# Generate data.
_X = np.arange(-1, 1, 0.05)
_Y = np.arange(-1, 1, 0.05)
X, Y = np.meshgrid(_X, _Y)
Z = -1*np.sqrt(4 - X**2 - Y**2)


# Plot contours
cont = plt.contourf(X, Y, Z)

# For update/animation
plt.ion()
for _, key in enumerate(_X):
    val = -1*math.sqrt(4-key**2-0)
    plt.scatter(0, key, color='r')
    plt.pause(0.05)
plt.close()
