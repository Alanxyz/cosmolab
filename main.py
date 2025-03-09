from pprint import pprint
from utils import *
from montecarlo import adjustparams

df = loaddf('jla_lcparams.txt')
adjustparams(df)
