# Adopted from https://matplotlib.org/gallery/subplots_axes_and_figures/subplot.html

import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-5.0, 5.0, num=200)

y_exp = np.exp(x)
y_square = np.square(x)
y_cube = np.power(x, 3)
y_log = np.log(x+5.1)

plt.subplot(2, 2, 1)
plt.plot(x, y_exp, 'o-', color='r')
plt.ylabel('exp(x)')
plt.xlabel('x')

plt.subplot(2, 2, 2)
plt.plot(x, y_square, '*-', color='g')
plt.xlabel('x')
plt.ylabel('square(x)')

plt.subplot(2, 2, 3)
plt.plot(x, y_cube, 'x-', color='b')
plt.xlabel('x')
plt.ylabel('cube(x)')


plt.subplot(2, 2, 4)
plt.plot(x, y_log, 'o--', color='m')
plt.xlabel('x')
plt.ylabel('log(x+5.1)')

plt.show()

 
