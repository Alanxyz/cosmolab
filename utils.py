import numpy as np

# Parameters
dx = 0.01
planck_matter_0 = 0.3111
planck_radiation_0 = 0.0000
planck_dark_energy_0 = 0.6889

# Constants
Gyr = 365 * 24 * 60**2 * 1e9
pc = 3.0857e16
G = 6.67e-11
c = 3e8
eV = 1.60e-19
critic_energy = 4870e6 * eV

# Math
fmap = lambda x, f: list(map(f, x))
delta = lambda a, b: 1 if a == b else 0

def loaddf(filename):
    file = f'dat/{filename}'
    with open(file, 'r') as f:
        firstline = f.readline().rstrip()
        headers = firstline[1:].split(' ')
        df = np.genfromtxt(file, dtype=None, names=headers)
        return df

def loadsqmat(filename):
    dat = np.loadtxt(f'dat/{filename}')
    n = int(dat[0])
    mat = dat[1:].reshape(n, n)
    return mat

