import matplotlib.pyplot as plt
from pprint import pprint

from utils import *
from plots import pltxy
from astropy.cosmology import Planck18 as cosmo
from montecarlo import adjustparams
from plots import triangle

df = loaddf('jla_lcparams.txt')
adjustparams(df)
