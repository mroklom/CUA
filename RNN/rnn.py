from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from matplotlib import pyplot
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, ParameterGrid, KFold
from loader import load_data_rnn
from sklearn.model_selection import train_test_split
import os
import numpy as np
from sklearn.metrics import f1_score

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def custom_f1_score(y_true, y_pred, n_decimals):
    # Convert the one hot encoded class vector to a single value class
    y_true_flat = np.argmax(y_true, axis=1)

    return np.round(f1_score(y_true_flat, y_pred, average='weighted'), decimals=n_decimals)


def create_model(n_units_output_lstm, n_units_output_dense, n_timesteps, n_features, n_classes):
    model = Sequential()
    model.add(LSTM(n_units_output_lstm, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(n_units_output_dense, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


def optimize_model(x, y, cv):
    n_timesteps = x.shape[1]
    n_features = x.shape[2]
    n_classes = y.shape[1]

    # Create the hyper parameters possible values
    batch_size = [50, 100]
    epochs = [5, 10]
    n_units_output_lstm = [100]
    n_units_output_dense = [100]

    # Create the hyper paramters grid
    param_grid = dict(
        batch_size=batch_size,
        epochs=epochs,
        n_units_output_lstm=n_units_output_lstm,
        n_units_output_dense=n_units_output_dense
    )

    possible_combination = list(ParameterGrid(param_grid))
    search_results = dict()

    for combination in possible_combination:
        print('Testing combination', str(combination))

        search_results[str(combination)] = []

        kf = KFold(n_splits=cv)

        # Cross validate results
        i = 1
        for train, test in kf.split(x):
            print('\tIteration', i)

            trainX = x[train]
            validationX = x[test]

            trainy = y[train]
            validationy = y[test]

            # Create the model
            model = create_model(
                n_units_output_lstm=combination['n_units_output_lstm'],
                n_units_output_dense=combination['n_units_output_dense'],
                n_timesteps=n_timesteps,
                n_features=n_features,
                n_classes=n_classes
            )

            # Train the model
            model.fit(trainX, trainy, batch_size=combination['batch_size'], epochs=combination['epochs'], verbose=2)

            # Make predictions
            predictions = model.predict_classes(validationX)

            # Assess and store performances
            search_results[str(combination)].append(custom_f1_score(y_true=validationy, y_pred=predictions, n_decimals=7))

            # increment iteration counter
            i = i + 1

        print('Results for', str(combination))
        print(search_results[str(combination)])
        print()

    # Find the best combination of parameters
    best_params_list = []
    max_f1 = 0
    for combination, f1 in search_results.items():
        mean_f1 = np.mean(f1)
        if mean_f1 >= max_f1:
            if mean_f1 > max_f1:
                best_params_list = [combination]
                max_f1 = mean_f1
            elif mean_f1 == max_f1:
                best_params_list.append(combination)

    return best_params_list, max_f1, search_results


def evaluate_emergency_vehicles():
    X, Y = load_data_rnn('/home/cua/PycharmProjects/CUA/Data/export-debut.csv')

    trainX, testX, trainy, testy = train_test_split(X, Y, test_size=0.25)
    n_timesteps, n_features, n_classes = trainX.shape[1], trainX.shape[2], trainy.shape[1]

    best_params_list, max_f1, search_results = optimize_model(trainX, trainy, cv=3)

    print('Best parameters possibilities found:\n', best_params_list, 'with performance:', max_f1)

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
    model.fit(trainX, trainy, batch_size=params['batch_size'], epochs=params['epochs'], verbose=1)

    # Assess the performances
    predictions = model.predict_classes(testX)
    print('y_true:', np.argmax(testy, axis=1))
    print('y_pred:', predictions)
    print()
    print('test f1 score weighted :', custom_f1_score(testy, predictions, n_decimals=7))


evaluate_emergency_vehicles()
