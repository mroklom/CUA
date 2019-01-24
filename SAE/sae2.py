from keras import Sequential, Input, Model
from keras.layers import Dense
from keras.utils import to_categorical

from loader import load_data_sae
import pandas as pd

import numpy as np


def create_sae(input_dim, structure, optimizer):
    input = Input(shape=(input_dim,))

    # Encoder part
    encoded = Dense(structure[0], activation='relu')(input)
    if structure.shape[0] > 1:
        for i in range(1, structure.shape[0]):
            encoded = Dense(structure[i])(encoded)

    # Decoder part
    decoded = Dense(structure[-2], activation='relu')(encoded)
    if structure.shape[0] > 2:
        for i in range(2, structure.shape[0]):
            decoded = Dense(structure[structure.shape[0] - (i + 1)], activation='relu')(decoded)

    decoded = Dense(input_dim, activation='relu')(decoded)

    autoencoder = Model(input, decoded)
    autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')

    encoder = Model(input, encoded)

    return autoencoder, encoder


def create_mlp(input_dim, n_classes, optimizer, structure=None):
    model = Sequential()

    if structure is not None:
        model.add(Dense(structure[0], activation='relu', input_dim=input_dim))
        if structure.shape[0] > 1:
            for i in range(1, structure.shape[0]):
                model.add(Dense(structure[i], activation='relu'))

        model.add(Dense(n_classes, activation='softmax'))

    else:
        model.add(Dense(n_classes, activation='relu', input_dim=input_dim))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


data = pd.DataFrame(load_data_sae('/home/cua/PycharmProjects/CUA/Data/export-debut.csv'))

X = data.drop(columns=['class'], axis=1).values
y = to_categorical(data['class'].values - 1, np.unique(data['class'].values).shape[0])

sae_structure = np.array([30, 20, 10])
mlp_structure = np.array([5])

sae, e = create_sae(X.shape[1], sae_structure, optimizer='adam')
print(sae.summary())

sae.fit(X, X, batch_size=10, epochs=12, verbose=2)

encodedX = e.predict(X)

mlp = create_mlp(input_dim=sae_structure[-1], structure=mlp_structure, n_classes=y.shape[1], optimizer='adam')
print(mlp.summary())

mlp.fit(encodedX, y, batch_size=10, epochs=42, verbose=2)


