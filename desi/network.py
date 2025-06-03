import numpy as np
from pprint import pprint

data = []
for section in ('0-100_000', '100_000-200_000'):
    datasection = np.load(f'galaxy_{section}.npy', allow_pickle=True)
    datasection = datasection[1:-1]
    data = np.concatenate((data, datasection))

data = [ datasect for datasect in data if datasect['subtype'] != '']
pprint(data)

