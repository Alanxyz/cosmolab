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

#print("FASE 1")
#d = getdata(0, 100_000)
#np.save('star_0-100_000', d)
print("FASE 3")
d = getdata(150_000, 200_000)
np.save('star_150_000-200_000', d)
