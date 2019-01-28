import pandas as pd
import numpy as np
import os

from keras.utils import to_categorical
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from Utils.sae_utils import optimize_sae, create_sae, optimize_mlp, create_mlp
from Utils.loader import load_data_sae

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data = pd.DataFrame(load_data_sae('/home/cua/PycharmProjects/CUA/Data/export-debut.csv'))

X = data.drop(columns=['class'], axis=1).values
y = to_categorical(data['class'].values - 1, np.unique(data['class'].values).shape[0])

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25)

# Optimize the sae
print('Optimizing SAE...\n')
params, loss, results = optimize_sae(train_x, cv=3)
best_params_sae = eval(params[0])
print('\nOptimizing SAE done!\n\n')

# Create the sae with it's best parameters and train it
print('Training SAE...\n')
sae, e = create_sae(input_dim=train_x.shape[1], structure=best_params_sae['structure'], optimizer='adam')
sae.fit(train_x, train_x, batch_size=best_params_sae['batch_size'], epochs=best_params_sae['epochs'], verbose=2)
print('Training SAE done!\n\n')

# Encode train X
encodedTrainX = e.predict(train_x)

# Optimize the mlp
print('Optimizing MLP...\n')
best_params_list, max_f1, search_results = optimize_mlp(train_x, train_y, cv=3)
best_params_mlp = eval(best_params_list[0])
print('Optimizing MLP done!\n\n')

# Create the sae with it's best parameters and train it
print('Training MLP...\n')
mlp = create_mlp(input_dim=encodedTrainX.shape[1], n_classes=train_y.shape[1], optimizer='adam', structure=best_params_mlp['structure'])
mlp.fit(encodedTrainX, train_y, batch_size=best_params_mlp['batch_size'], epochs=best_params_mlp['epochs'], verbose=2)
print('Training MLP done!\n\n')

# Evaluate the model
encodedTestX = e.predict(test_x)
predictions = mlp.predict_classes(encodedTestX)
print('y_true:', np.argmax(test_y, axis=1))
print('y_pred:', predictions)
print()
print('test f1 score weighted :', np.round(
    f1_score(y_true=np.argmax(test_y, axis=1), y_pred=predictions, average='weighted'),
    decimals=7
))
