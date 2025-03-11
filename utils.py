import numpy as np

# Math
fmap = lambda x, f: list(map(f, x))
delta = lambda a, b: 1 if a == b else 0

def loaddf(filename):
    with open(filename, 'r') as f:
        firstline = f.readline().rstrip()
        headers = firstline[1:].split(' ')
        df = np.genfromtxt(filename, dtype=None, names=headers)
        return df

def loadsqmat(filename):
    dat = np.loadtxt(filename)
    n = int(dat[0])
    mat = dat[1:].reshape(n, n)
    return mat

