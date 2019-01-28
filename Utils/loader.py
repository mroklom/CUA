import pandas as pd
import numpy as np

from keras.utils import to_categorical


def load_data_rnn(path, pad=True):
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

    if pad:
        max_l = get_max_length_sequence(values)

        print('max length:', max_l)

        print('Shapes before padding:', values.shape, 'first trajectory:', values[0].shape)
        values = preprocess_sequences(values, max_l, values[0].shape[1])
        print('Shapes after padding:', values.shape, 'first trajectory:', values[0].shape)
        print()

    return values, np.array(classes)


def load_data_sae(path):
    df = pd.read_csv(path, sep=';', header=0)

    column_list_to_remove = [
        'id_vehicule',
        'debut_traj',
        'fin_traj',
        'id_ligne',
        'DATE'
    ]

    df.drop(columns=column_list_to_remove, inplace=True, axis=1)

    mapping = dict(
        RPM='real',
        SPEED='real',
        RIGHT_FLASH='boolean',
        LEFT_FLASH='boolean',
        BRAKE_PEDAL='boolean',
        PRIMARY_LT='boolean',
        WIG_WAG='boolean',
        REVERSE_GEAR='boolean',
        CHASSIS_VOLT='real',
        CONVERS_VOLT='real',
        PARKING_LT='boolean',
        HIGH_BEAM_LT='boolean',
        SIREN='boolean',
        THROTTLE='real',
        ACC_LONG='real',
        ACC_LATERAL='real',
        ACC_VERTICAL='real',
        FUEL_RATE='real',
        DRIVER_DOOR='boolean',
        BACK_DOOR='boolean',
        PARK_BRAKE='boolean',
        SEAT_BELT='boolean'
    )

    grouped = df.groupby(['id_traj'])
    data = dict()
    for col in mapping:
        if mapping[col] == 'real':
            data[col + '_mean'] = []
            data[col + '_std'] = []
            data[col + '_max'] = []
            data[col + '_min'] = []
        else:
            data[col + '_uptime'] = []
    data['class'] = []

    for name, group in grouped:
        if name == 538:
            print('exception on', name, 'with', len(group), 'points')
        else:
            data['class'].append(group['classe'].max())
            # Compute features for each trajectory
            for col in mapping:

                # If the series is real, compute mean etc ..
                if mapping[col] == 'real':
                    data[col + '_mean'].append(group[col].mean())
                    data[col + '_std'].append(group[col].std())
                    data[col + '_max'].append(group[col].max())
                    data[col + '_min'].append(group[col].min())

                # If the series is boolean, compute ration of true over false
                elif mapping[col] == 'boolean':
                    data[col + '_uptime'].append(group[col].mean())

    return data


def get_max_length_sequence(data):
    max_length = 0
    for i in range(len(data)):
        if data[i].shape[0] > max_length:
            max_length = data[i].shape[0]

    return max_length


def preprocess_sequences(data, max_length, n_features):
    new_data = []

    for sequence in data:

        new_sequence = np.array(sequence[0:sequence.shape[0]])

        diff_length = max_length - sequence.shape[0]
        if diff_length != 0:
            for i in range(diff_length):
                new_sequence = np.concatenate((new_sequence, np.array([[0 for i in range(n_features)]])))

        new_data.append(new_sequence)

    return np.array(new_data)
