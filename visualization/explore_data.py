import pandas as pd
import matplotlib.pyplot as plt

classes_names = [
    'Vers intervention',
    'Déplacement intra POI',
    'vers hopital',
    'retour caserne',
    'retour logistique',
    'autre'
]

df = pd.read_csv('/home/cua/PycharmProjects/CUA/Data/export-debut.csv', sep=';', header=0)

column_list_to_remove = [
    'id_vehicule',
    'debut_traj',
    'fin_traj',
    'id_ligne',
    'DATE'
]

df.drop(columns=column_list_to_remove, inplace=True, axis=1)

# Proportion of classes
# Number of point per trajectory
# Number of point per classes

traj_group = df.groupby(['id_traj'])

new_df_dict = dict()

new_df_dict['traj'] = []
new_df_dict['class'] = []
new_df_dict['length'] = []

for name, group in traj_group:
    new_df_dict['traj'].append(name)
    new_df_dict['class'].append(group['classe'].max())
    new_df_dict['length'].append(group.shape[0])

new_df = pd.DataFrame(new_df_dict)

group_class = new_df.groupby(['class'])

ax = plt.subplot()
ax.boxplot(new_df.length.values)
ax.set_yscale('log')
plt.show()
