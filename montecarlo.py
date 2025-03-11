from multiprocessing import Pool
import numpy as np
import emcee
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from utils import *

ceta = np.load('out/cstat.npy')
c00 = loadsqmat('dat/jla_v0_covmatrix.dat')
c11 = loadsqmat('dat/jla_va_covmatrix.dat')
c22 = loadsqmat('dat/jla_vb_covmatrix.dat')
c01 = loadsqmat('dat/jla_v0a_covmatrix.dat')
c02 = loadsqmat('dat/jla_v0b_covmatrix.dat')
c12 = loadsqmat('dat/jla_vab_covmatrix.dat')
n = 740

def log_likelihood(th, x, zdat, yerr):
    alpha, beta, mb1, dm, Om0 = th

    cosmo = FlatLambdaCDM(H0=70, Om0=Om0, Tcmb0=2.725)
    distlum = cosmo.luminosity_distance(zdat).to_value(u.pc)
    mudat = 5 * np.log10(distlum) - 5

    eta = np.column_stack((x['mb'], x['x1'], x['color'])).ravel()
    id = np.identity(n)
    a = np.tensordot(id, np.array([ 1, alpha, -beta ]), axes = 0).reshape(n, 3 * n)

    mb = np.ones(n) * mb1
    mb[x['3rdvar'] >= 10] += dm

    c = a @ ceta @ a.T
    mu = a @ eta - mb
    res = mu - mudat

    cinv = np.linalg.inv(c)
    xi2 = res.T @ cinv @ res

    return -0.5 * xi2

def log_slikelihood(th, x, zdat, yerr):
    alpha, beta, mb1, dm, Om0 = th

    # Data
    cosmo = FlatLambdaCDM(H0=70, Om0=Om0, Tcmb0=2.725)
    distlum = cosmo.luminosity_distance(zdat).to_value(u.pc)
    mudat = 5 * np.log10(distlum) - 5


    # Model
    mb = np.ones(n) * mb1
    mb[x['3rdvar'] >= 10] += dm
    mu = x['mb'] - (mb + alpha * x['x1'] - beta * x['color'])


    # Comparation
    res = mu - mudat

    cov = c00  \
        + alpha**2 * c11 \
        + beta**2 * c22 \
        + 2 * alpha * c01 \
        + -2 * beta * c02 \
        + -2 * alpha * beta * c12

    ddiag = x['dmb']**2 \
        + (alpha * x['dx1'])**2 \
        + (beta * x['dcolor'])**2 \
        + 2 * alpha * x['cov_m_s'] \
        - 2 * beta * x['cov_m_c'] \
        - 2 * alpha * beta * x['cov_s_c']

    cov += np.diagflat(ddiag)

    covinv = np.linalg.inv(cov)
    xi2 = res.T @ covinv @ res
    return -xi2

def log_prior(th):
    alpha, beta, mb1, dm, Om0 = th
    if -1.0 < alpha < 1.0 and 0.0 < beta < 5.0 and -1.0 < dm < 1.0 and -20 < mb1 < -18 and 0.2 < Om0 < 0.4 :
        return 0.0
    else:
        return -np.inf

def log_probability(th, x, y, yerr):
    lp = log_prior(th)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_slikelihood(th, x, y, yerr)

dim = 5
walkers = 2 * dim
pos = np.zeros((walkers, dim))

mean = [0.140, 3.139, -19.04, -0.060, 0.289]
for i in range(dim):
    for j in range(walkers):
        pos[j, i] = np.random.normal(mean[i], 0.001)

data = loaddf('dat/jla_lcparams.txt')

with Pool() as pool:
    sampler = emcee.EnsembleSampler(
        walkers,
        dim,
        log_probability,
        args=[ data, data['zcmb'], data['dz'] ],
        pool=pool
    )
    sampler.run_mcmc(pos, 1000, progress=True);

chain = sampler.get_chain(flat=True)
np.save("out/chain", chain)
