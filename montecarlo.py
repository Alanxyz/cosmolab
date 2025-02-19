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

def log_likelihood(th, x, yz, yerr):
    alpha, beta, mb1, dm = th
    shape, color, mstellar = x['x1'], x['color'], x['mb']

    n = len(x['x1'])
    model = getmodel(alpha, beta, mb1, dm)
    y_model = model(mstellar, shape, color)
    distlum = cosmo.luminosity_distance(yz)

    mb = x['mb']
    eta = np.column_stack((x['mb'], x['x1'], x['color'])).ravel()
    cstat = c00 @ np.ones(3, 3)
    ceta = cstat
    id = np.identity(n)
    a = np.tensordot(id, np.array([ 1, alpha, -beta ]))
    c = a @ ceta @ a.T
    y = 5 * np.log(distlum / (10 * u.pc)) - 5

    mu = a @ eta - mb

    return -0.5 * np.sum(chi2 + np.log(sigma2))

def log_prior(th):
    alpha, beta, mb1, dm = th
    return 0.0

def log_probability(th, x, y, yerr):
    lp = log_prior(th)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(th, x, y, yerr)

def adjustparams(df):
    dim = 4
    walkers = 8
    pos = 0.5 + 1 * np.random.randn(walkers, dim)

    data = loaddf('jla_lcparams.txt')

    sampler = emcee.EnsembleSampler(
        walkers,
        dim,
        log_probability,
        args=[ data, data['zcmb'], data['dz'] ]
    )

    sampler.run_mcmc(pos, 2000, progress=True);
    return sampler
