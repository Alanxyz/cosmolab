import os
import sys
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.convolution import convolve, Gaussian1DKernel
from sparcl.client import SparclClient

table = Table.read('dat/zall-pix-fuji.fits')
table['HEALPIX']
mask = table['SPECTYPE'] == 'STAR'
table = table[mask]
targetid = table['TARGETID']
x = list(targetid)
x = [ int(y) for y in x ]
client = SparclClient()

def getdata(a, b):
    inc = ['specid', 'flux', 'wavelength', 'redshift', 'subtype']
    res = client.retrieve_by_specid(specid_list = x[a:b], include = inc, dataset_list = ['DESI-DR1'])
    return res.data

for n in range(20):
    print(f'FASE {n}')
    a = 50_000 * n
    b = a + 49_999
    d = getdata(a, b)
    np.save(f'star_{a}-{b}', d)
