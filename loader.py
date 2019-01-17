import pandas as pd
import numpy as np

prefix = '/users/21509823t/PycharmProjects/CUA/Data/'


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
        # Not so clean but anyway
        classes.append(group['classe'].max())

        # Append everything except id_traj and classe
        values.append(np.delete(group.values, [0, 1], 1))

    return np.array(values), np.array(classes)
