import numpy as np
from pprint import pprint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = []
sectsize = 50_000
sectnames = [
    f'{sectsize * n}-{sectsize * n + sectsize - 1}' for n in range(20)
]
for section in sectnames:
    datasection = np.load(f'star_{section}.npy', allow_pickle=True)
    datasection = datasection[1:-1]
    data = np.concatenate((data, datasection))

startypes = ['O', 'B', 'A', 'F', 'G', 'K', 'M']

data = [
    datasect for datasect in data
    if datasect['subtype'] in startypes
]
data = np.array(data)

fluxes = [d['flux'] for d in data]
labels = [d['subtype'] for d in data]

target_len = max(len(f) for f in fluxes)
X = np.array([
    np.pad(f, (0, target_len - len(f)), mode='constant') for f in fluxes
])
X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
X = X[..., np.newaxis]

le = LabelEncoder()
y = le.fit_transform(labels)

xtrain, xtest, ytrain, ytest = train_test_split(
    X,
    y,
    stratify=y,
    test_size=0.2,
    random_state=42
)

from models import *

model = buildmodel1(inputlength=X.shape[1], numclasses=len(le.classes_))
history = model.fit(
    xtrain,
    ytrain,
    validation_data=(xtest, ytest),
    epochs=4,
    batch_size=64
)

yest = np.argmax(model.predict(xtest), axis=1)
print(classification_report(ytest, yest, target_names=le.classes_))
