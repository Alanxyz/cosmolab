import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from model import getmodel
from utils import *
from pprint import pprint

c00 = loadsqmat('jla_v0_covmatrix.dat')
c11 = loadsqmat('jla_va_covmatrix.dat')
c22 = loadsqmat('jla_vb_covmatrix.dat')
c01 = loadsqmat('jla_v0a_covmatrix.dat')
c02 = loadsqmat('jla_v0b_covmatrix.dat')
c12 = loadsqmat('jla_vab_covmatrix.dat')

cstat = np.block([
    [c01, c01, c02],
    [c01, c11, c12],
    [c02, c12, c22]
])

n = 740
cstat = np.zeros((3 * n, 3 * n))

for i in range(n):
    for j in range(n):
        cstat[3*i, 3*j] = c00[i, j]
        cstat[3*i + 1, 3*j + 1] = c11[i, j]
        cstat[3*i + 2, 3*j + 2] = c22[i, j]

        cstat[3*i + 1, 3*j] = c01[i, j]
        cstat[3*i, 3*j+1] = c01[i, j]

        cstat[3*i + 2, 3*j] = c02[i, j]
        cstat[3*i, 3*j+2] = c02[i, j]

        cstat[3*i + 1, 3*j + 2] = c12[i, j]
        cstat[3*i + 2, 3*j+1] = c12[i, j]
#matplotlib.use('PDF')
#plt.imshow(np.log(cstat), cmap='hot', interpolation='nearest')
#plt.savefig('plt/hot.png')
np.save("cov.txt", cstat)
