import numpy as np
from pprint import pprint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Dense, Flatten, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam

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
    if datasect['subtype'] in startypes]
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

def buildmodel(inputlength, numclasses):
    model = Sequential([
        Conv1D(32, 7, activation='relu', input_shape=(inputlength, 1)),
        MaxPooling1D(2),
        BatchNormalization(),

        Conv1D(64, 5, activation='relu'),
        MaxPooling1D(2),
        BatchNormalization(),

        Conv1D(128, 3, activation='relu'),
        MaxPooling1D(2),
        BatchNormalization(),

        Flatten(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(numclasses, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = buildmodel(inputlength=X.shape[1], numclasses=len(le.classes_))
history = model.fit(
    xtrain,
    ytrain,
    validation_data=(xtest, ytest),
    epochs=20,
    batch_size=64
)

yest = np.argmax(model.predict(xtest), axis=1)
print(classification_report(ytest, yest, target_names=le.classes_))
