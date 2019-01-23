from keras import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

from loader import load_data_sae
import pandas as pd

import numpy as np

data = pd.DataFrame(load_data_sae('/home/cua/PycharmProjects/CUA/Data/export-debut.csv'))

X = data.drop(columns=['class'], axis=1).values
y = to_categorical(data['class'].values - 1, np.unique(data['class'].values).shape[0])

print(y)
print()

model = Sequential()
model.add(Dense(2, activation='relu', input_dim=X.shape[1]))
model.add(Dense(X.shape[1], activation='relu'))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X, X, batch_size=50, epochs=1000, verbose=2)
