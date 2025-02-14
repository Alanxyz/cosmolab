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
    model = getmodel(alpha, beta, mb1, dm)
    y_model = model(mstellar, shape, color)
    distlum = cosmo.luminosity_distance(yz)
    y = 5 * np.log(distlum / (10 * u.pc)) - 5

    cov = c00
    cov += alpha**2 * c11
    cov += beta**2 * c22
    cov += 2 * alpha * c01
    cov += -2 * beta * c02
    cov += -2 * alpha * beta * c12

    ddiag = x['dmb']**2 + (alpha * x['dx1'])**2 + (beta * x['dcolor'])**2 + 2 * alpha * x['cov_m_s'] - 2 * beta * x['cov_m_c'] - 2 * alpha * beta * x['cov_s_c']
    n = 740
    for i in range(n):
        cov[i][i] += ddiag[i]

    np.linalg.cholesky(cov)

    sigma2 = yerr**2 + y_model**2
    chi2 = (y - y_model) ** 2 / sigma2

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
