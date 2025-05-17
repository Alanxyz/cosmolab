Los siguientes códigos están pensados para ser ejecutados en un servidor local. Primero hay que entrar al servidor

```
pass dciserver -c
ssh 148.214.16.7
conda activate desi
```


# Setup

Necesitaremos de algunas librerias


```bash
pip install fitsio
wget https://github.com/desihub/tutorials/blob/main/mpl/desi.mplstyle
```


```
import os
import numpy as np

from astropy.io import fits
from astropy.table import Table
from astropy.convolution import convolve, Gaussian1DKernel
```
