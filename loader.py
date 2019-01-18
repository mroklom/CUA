import pandas as pd
import numpy as np
from keras.utils import to_categorical
from test import get_max_length_sequence, preprocess_sequences


def load_data(path):
    df = pd.read_csv(path, sep=';', header=0)

    column_list_to_remove = [
        'id_vehicule',
        'debut_traj',
        'fin_traj',
        'id_ligne',
        'DATE'
    ]

    df.drop(columns=column_list_to_remove, inplace=True, axis=1)

    grouped = df.groupby(['id_traj'])
    values = []
    classes = []

    for name, group in grouped:

        # Append everything except id_traj and classe
        if name == 538:
            print('exception on', name, 'with', len(group), 'points')
        else:
            values.append(np.delete(group.values, [0, 1], 1))
            # Not so clean but anyway
            classes.append(to_categorical(group['classe'].max() - 1, 6))

    values = np.array(values)

    max_l = get_max_length_sequence(values)

    print('max length:', max_l)

    print('Shapes before padding:', values.shape, 'first trajectory:', values[0].shape)
    values = preprocess_sequences(values, max_l, values[0].shape[1])
    print('Shapes after padding:', values.shape, 'first trajectory:', values[0].shape)
    print()

    return values, np.array(classes)


def return_observation_by_observation(X, y):
    i = 0
    while True:
        yield np.array([X[i]]), np.array([y[i]])

        i = i + 1

        if i > X.shape[0] - 1:
            i = 0
