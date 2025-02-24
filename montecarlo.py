import numpy as np
import emcee
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

from model import getmodel
from utils import *

ceta = np.load("out/cstat.npy")
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
    alpha, beta, mb1, dm = th
    shape, color, mstellar = x['x1'], x['color'], x['mb']
    model = getmodel(alpha, beta, mb1, dm)


    mb = mb1
    y_model = mbstar - (mb - alpha * shape + beta * color)

    distlum = cosmo.luminosity_distance(zdat)
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

    sigma2 = yerr**2 + y_model**2
    chi2 = (y - y_model) ** 2 / sigma2

    return -0.5 * np.sum(chi2 + np.log(sigma2))

def log_prior(th):
    alpha, beta, mb1, dm, Om0 = th
    if 0.0 < alpha < 0.2 and 2.5 < beta < 3.5 and -0.2 < dm < 0.1 and -20 < mb1 < -18 and 0.2 < Om0 < 0.4 :
        return 0.0
    else:
        return -np.inf

def log_probability(th, x, y, yerr):
    lp = log_prior(th)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(th, x, y, yerr)

def adjustparams(df):
    dim = 5
    walkers = 2 * dim
    pos = np.zeros((walkers, dim))

    mean = [0, 3, -19, -0.15, 0.29]
    for i in range(dim):
        for j in range(walkers):
            pos[j, i] = np.random.normal(mean[i], 0.01)

    data = loaddf('jla_lcparams.txt')

    sampler = emcee.EnsembleSampler(
        walkers,
        dim,
        log_probability,
        args=[ data, data['zcmb'], data['dz'] ]
    )

    sampler.run_mcmc(pos, 1000, progress=True);
    chain = sampler.get_chain(flat=True)
    np.save("out/chain", chain)
