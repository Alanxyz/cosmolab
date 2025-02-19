import numpy as np
import emcee
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

from model import getmodel
from utils import *

c00 = loadsqmat('jla_v0_covmatrix.dat')
c11 = loadsqmat('jla_va_covmatrix.dat')
c22 = loadsqmat('jla_vb_covmatrix.dat')
c01 = loadsqmat('jla_v0a_covmatrix.dat')
c02 = loadsqmat('jla_v0b_covmatrix.dat')
c12 = loadsqmat('jla_vab_covmatrix.dat')

def log_likelihood(th, x, zdat, yerr):
    alpha, beta, mb1, dm = th
    shape, color, mstellar = x['x1'], x['color'], x['mb']

    n = len(x['x1'])
    model = getmodel(alpha, beta, mb1, dm)

    distlum = cosmo.luminosity_distance(zdat)
    mudat = 5 * np.log(distlum / (10 * u.pc)) - 5

    mb = x['mb']
    eta = np.column_stack((x['mb'], x['x1'], x['color'])).ravel()

    cstat = np.block([
        [c00, c01, c02],
        [c01, c11, c12],
        [c02, c12, c22]
    ])
    ceta = cstat

    id = np.identity(n)
    a = np.tensordot(id, np.array([ 1, alpha, -beta ]), axes = 0).reshape(n, 3 * n)

    c = a @ ceta @ a.T
    mu = a @ eta - mb
    res = mu - mudat
    xi = mu @ c @ mu.T

    return -0.5 * xi

def log_slikelihood(th, x, zdat, yerr):
    alpha, beta, mb1, dm = th
    shape, color, mstellar = x['x1'], x['color'], x['mb']

    n = len(x['x1'])
    model = getmodel(alpha, beta, mb1, dm)

    distlum = cosmo.luminosity_distance(zdat)
    mudat = 5 * np.log(distlum / (10 * u.pc)) - 5

    mb = x['mb']
    eta = np.column_stack((x['mb'], x['x1'], x['color'])).ravel()

    cstat = np.block([
        [c00, c01, c02],
        [c01, c11, c12],
        [c02, c12, c22]
    ])
    ceta = cstat

    id = np.identity(n)
    a = np.tensordot(id, np.array([ 1, alpha, -beta ]), axes = 0).reshape(n, 3 * n)

    c = a @ ceta @ a.T
    mu = a @ eta - mb
    res = mu - mudat
    xi = mu @ c @ mu.T

    return -0.5 * xi

def log_prior(th):
    alpha, beta, mb1, dm = th
    if 0.0 < alpha < 0.2 and 2.5 < beta < 3.5 and -0.2 < dm < 0.1 and -20 < mb1 < -18:
        return 0.0
    else:
        return -np.inf

def log_probability(th, x, y, yerr):
    lp = log_prior(th)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(th, x, y, yerr)

def adjustparams(df):
    dim = 4
    walkers = 8
    pos = np.zeros((walkers, dim))

    mean = [0, 3, -19, -0.15]
    for i in range(4):
        for j in range(8):
            pos[j, i] = np.random.normal(mean[i], 0.01)

    print(pos)

    data = loaddf('jla_lcparams.txt')

    sampler = emcee.EnsembleSampler(
        walkers,
        dim,
        log_probability,
        args=[ data, data['zcmb'], data['dz'] ]
    )

    sampler.run_mcmc(pos, 200, progress=True);
    return sampler
