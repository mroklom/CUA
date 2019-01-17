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

    for name, group in grouped:
        # print(name)
        # print(group.values)

        values.append(group.values)

    return np.array(values)


data = load_data(prefix + 'export-debut.csv')

for trajectory in data:
    print(trajectory.shape)
