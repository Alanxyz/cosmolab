import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils import *
from pprint import pprint


fits = np.load("cetafits.npy")
c00 = loadsqmat('jla_v0_covmatrix.dat')
c11 = loadsqmat('jla_va_covmatrix.dat')
c22 = loadsqmat('jla_vb_covmatrix.dat')
c01 = loadsqmat('jla_v0a_covmatrix.dat')
c02 = loadsqmat('jla_v0b_covmatrix.dat')
c12 = loadsqmat('jla_vab_covmatrix.dat')

n = 740
c = np.zeros((3 * n, 3 * n))

for i in range(n):
    for j in range(n):
        c[3 * i + 2, 3 * j + 2] = c00[i, j]
        c[3 * i + 1, 3 * j + 1] = c11[i, j]
        c[3 * i, 3 * j] = c22[i, j]

        c[3 * i + 2, 3 * j + 1] = c01[i, j]
        c[3 * i + 2, 3 * j] = c02[i, j]
        c[3 * i, 3 * j + 1] = c12[i, j]

        c[3 * j + 1, 3 * i + 2] = c01[i, j]
        c[3 * j, 3 * i + 2] = c02[i, j]
        c[3 * j + 1, 3 * i] = c12[i, j]

plt.imshow(np.log(c/fits), cmap='hot', interpolation='nearest')
plt.show()
#np.save("ceta", c)
