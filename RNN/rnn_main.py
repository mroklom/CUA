import os
import numpy as np

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from Utils.loader import load_data_rnn
from Utils.rnn_utils import optimize_model, create_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

X, y = load_data_rnn('/home/cua/PycharmProjects/CUA/Data/export-debut.csv')

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25)

n_timesteps = train_x.shape[1]
n_features = train_x.shape[2]
n_classes = train_y.shape[1]

best_params_list, max_f1, search_results = optimize_model(train_x, train_y, cv=3)

print('Best parameters possibilities found:\n', best_params_list, 'with f1_score of', max_f1)

# Ideally pick the simplest model in order to better generalize
params = eval(best_params_list[0])

# Create the best model found
model = create_model(
    n_units_output_lstm=params['n_units_output_lstm'],
    n_units_output_dense=params['n_units_output_dense'],
    n_timesteps=n_timesteps,
    n_features=n_features,
    n_classes=n_classes
)

print(model.summary())

# Fit the model
model.fit(train_x, train_y, batch_size=params['batch_size'], epochs=params['epochs'], verbose=2)

# Assess the performances
predictions = model.predict_classes(test_x)
print('y_true:', np.argmax(test_y, axis=1))
print('y_pred:', predictions)
print()
print('test f1 score weighted :', np.round(
    f1_score(y_true=np.argmax(test_y, axis=1), y_pred=predictions, average='weighted'),
    decimals=7
))
