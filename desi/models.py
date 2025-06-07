from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, Dense, Flatten, Dropout, BatchNormalization
)

def buildmodel1(inputlength, numclasses):
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

def buildmodel0(inputlength, numclasses):
    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=(inputlength, 1)),
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

def buildmodel2(inputlength, numclasses):
    model = Sequential([
        Conv1D(16, 7, activation='relu', input_shape=(inputlength, 1)),
        MaxPooling1D(2),
        BatchNormalization(),

        Conv1D(16, 7, activation='relu'),
        MaxPooling1D(2),
        BatchNormalization(),

        Conv1D(16, 7, activation='relu'),
        MaxPooling1D(2),
        BatchNormalization(),

        Conv1D(16, 7, activation='relu'),
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
