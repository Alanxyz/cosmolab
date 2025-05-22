Los siguientes códigos están pensados para ser ejecutados en un servidor local. Primero hay que entrar al servidor

# Setup

Necesitaremos de algunas librerias

```bash
pip install fitsio
git clone https://github.com/desihub/desitarget.git --depth=1
git clone https://github.com/desihub/desiutil.git --depth=1
git clone https://github.com/desihub/desispec.git --depth=1
git clone https://github.com/desihub/desimodel.git --depth=1
git clone https://github.com/desihub/speclite.git --depth=1
```

```python
import os
import sys
import numpy as np

from astropy.io import fits
from astropy.table import Table
from astropy.convolution import convolve, Gaussian1DKernel
```

```python
mypath = '/work/alan/projecta/desi'
sys.path.insert(1, f'{mypath}/desitarget/py/')
sys.path.insert(1, f'{mypath}/desiutil/py/')
sys.path.insert(1, f'{mypath}/desispec/py/')
sys.path.insert(1, f'{mypath}/desimodel/py/')
sys.path.insert(1, f'{mypath}/speclite/')

from desimodel.footprint import radec2pix
import desispec.io
from desispec import coaddition

from desitarget.sv1 import sv1_targetmask    # For SV1
from desitarget.sv2 import sv2_targetmask    # For SV2
from desitarget.sv3 import sv3_targetmask    # For SV3
```

```

```

```python
from sparcl.client import SparclClient

client = SparclClient()
inc = ['specid', 'flux', 'wavelength','targetid']

res = client.retrieve_by_specid(specid_list = x, include = inc, dataset_list = ['DESI-DR1'])

table = Table.read('fuji/zpix-sv3-bright.fits')
mask = table['SPECTYPE'] == 'GALAXY'
targetid = table[mask]['TARGETID']
x = list(targetid)
x = [ int(y) for y in x ]
table.columns
```
