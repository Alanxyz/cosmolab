Los siguientes códigos están pensados para ser ejecutados en un servidor local. Primero hay que entrar al servidor

# Setup

Cosasas de machine learning:

```bash
conda create -n ml-env python=3.11 tensorflow keras scikit-learn astropy fitsio -c conda-forge
pip install sparclclient
```

El paquete más problematico es tensorflow, hay que revisar que se haya instalado correctamente.

```python
import tensorflow
```

Necesitaremos de algunas librerias propias de DESI

```bash
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

# Carga de datos

La descarga de datos será pediente Sparcl. Debo optimizar esta sección.

```python
from sparcl.client import SparclClient

table = Table.read('fuji/zpix-sv3-bright.fits')
mask = table['SPECTYPE'] == 'GALAXY'

targetid = table[mask]['TARGETID']
x = list(targetid)
x = [ int(y) for y in x ]
table.columns
client = SparclClient()

def getdata(a, b):
    inc = ['specid', 'flux', 'wavelength', 'redshift']
    res = client.retrieve_by_specid(specid_list = x[a:b], include = inc, dataset_list = ['DESI-DR1'])
    return res.data

getdata(4, 9)
```

# Machine Learning


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Supón que tus datos están así:
# X.shape = (n_samples, 3600), Y.shape = (n_samples,) con clases como 0 = espiral, 1 = elíptica, etc.

X = np.random.rand(1000, 3600).astype(np.float32)
Y = np.random.randint(0, 3, 1000)  # 3 clases

X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-6)
X = X[..., np.newaxis]  # (samples, 3600, 1) para Conv1D
Y_cat = to_categorical(Y)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y_cat, test_size=0.2)

model = Sequential([
    Conv1D(32, kernel_size=5, activation='relu', input_shape=(3600, 1)),
    MaxPooling1D(pool_size=2),
    BatchNormalization(),

    Conv1D(64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    BatchNormalization(),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(Y_cat.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=20, batch_size=32)
```
