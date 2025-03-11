import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import pypolychord as pc
from pypolychord.settings import PolyChordSettings
from utils import *

c00 = loadsqmat('dat/jla_v0_covmatrix.dat')
c11 = loadsqmat('dat/jla_va_covmatrix.dat')
c22 = loadsqmat('dat/jla_vb_covmatrix.dat')
c01 = loadsqmat('dat/jla_v0a_covmatrix.dat')
c02 = loadsqmat('dat/jla_v0b_covmatrix.dat')
c12 = loadsqmat('dat/jla_vab_covmatrix.dat')
n = 740

data = loaddf('dat/jla_lcparams.txt')

def loglikelihood(th):
    alpha, beta, mb1, dm, Om0 = th
    
    cosmo = FlatLambdaCDM(H0=70, Om0=Om0, Tcmb0=2.725)
    distlum = cosmo.luminosity_distance(data['zcmb']).to_value(u.pc)
    mudat = 5 * np.log10(distlum) - 5
    
    mb = np.ones(n) * mb1
    mb[data['3rdvar'] >= 10] += dm
    mu = data['mb'] - (mb + alpha * data['x1'] - beta * data['color'])
    res = mu - mudat
    
    cov = c00 \
          + alpha**2 * c11 \
          + beta**2 * c22 \
          + 2 * alpha * c01 \
          - 2 * beta * c02 \
          - 2 * alpha * beta * c12

    ddiag = (data['dmb'])**2  \
            + (alpha * data['dx1'])**2 \
            + (beta * data['dcolor'])**2 \
            + 2 * alpha * data['cov_m_s'] \
            - 2 * beta * data['cov_m_c'] \
            - 2 * alpha * beta * data['cov_s_c']

    cov += np.diagflat(ddiag)
    covinv = np.linalg.inv(cov)
    xi2 = res.T @ covinv @ res
    
    return -0.5 * xi2

def prior(unit_cube):
    alpha = -1.0 + 2.0 * unit_cube[0]
    beta = 5.0 * unit_cube[1]
    mb1 = -20.0 + 2.0 * unit_cube[2]
    dm = -1.0 + 2.0 * unit_cube[3]
    Om0 = 0.2 + 0.2 * unit_cube[4]
    
    return [alpha, beta, mb1, dm, Om0]

ndim = 5
settings = PolyChordSettings(ndim, 0)
settings.file_root = 'jla_polychord'
settings.nlive = 100
settings.num_repeats = 1000
settings.do_clustering = True

# Ejecutar PolyChord
pc.run_polychord(loglikelihood, ndim, 0, settings, prior=prior)
