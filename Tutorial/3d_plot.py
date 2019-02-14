# Adopted from: https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import math

fig = plt.figure()
ax = fig.gca(projection='3d')

# Generate data.
_X = np.arange(-1, 1, 0.05)
_Y = np.arange(-1, 1, 0.05)
X, Y = np.meshgrid(_X, _Y)
Z = -1*np.sqrt(4 - X**2 - Y**2)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

#For update/animation
plt.ion()
for _, key in enumerate(_X):
    val = -1*math.sqrt(4-key**2-1)
    ax.scatter(1, key, val, color='r')
    # Pause for better visualization
    plt.pause(0.1)
plt.close()
