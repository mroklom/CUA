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
from sklearn.model_selection import GridSearchCV
from loader import load_data_rnn
from sklearn.model_selection import train_test_split
import os
import numpy as np

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def create_model(n_units_output_lstm, n_units_output_dense, n_timesteps, n_features, n_classes):
    model = Sequential()
    model.add(LSTM(n_units_output_lstm, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(n_units_output_dense, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def optimize_model():
    # Load  the data
    X, Y = load_data_rnn('/home/cua/PycharmProjects/CUA/Data/export-debut.csv')

    trainX, testX, trainy, testy = train_test_split(X, Y, test_size=0.25)

    n_timesteps, n_features, n_classes = trainX.shape[1], trainX.shape[2], trainy.shape[1]

    # Map the model to scikit learn
    model = KerasClassifier(
        build_fn=create_model,
        n_timesteps=n_timesteps,
        n_features=n_features,
        n_classes=n_classes,
        verbose=0
    )

    # Create the hyper parameters possible values
    batch_size = [100]
    epochs = [10]
    n_units_output_lstm = [100]
    n_units_output_dense = [100]

    # Create the hyper paramters grid
    param_grid = dict(
        batch_size=batch_size,
        epochs=epochs,
        n_units_output_lstm=n_units_output_lstm,
        n_units_output_dense=n_units_output_dense
    )

    # Initialize the grid search
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        n_jobs=-1,
        cv=2,
        verbose=2
    )

    grid_result = grid.fit(trainX, trainy, verbose=2)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    return grid_result.best_params_


# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


def evaluate_emergency_vehicles():
    X, Y = load_data_rnn('/home/cua/PycharmProjects/CUA/Data/export-debut.csv')

    trainX, testX, trainy, testy = train_test_split(X, Y, test_size=0.25)
    n_timesteps, n_features, n_classes = trainX.shape[1], trainX.shape[2], trainy.shape[1]

    params = optimize_model()

    model = create_model(
        n_units_output_lstm=params['n_units_output_lstm'],
        n_units_output_dense=params['n_units_output_dense'],
        n_timesteps=n_timesteps,
        n_features=n_features,
        n_classes=n_classes
    )

    print(model.summary())

    model.fit(trainX, trainy, batch_size=params['batch_size'], epochs=params['epochs'], verbose=1)
    loss, accuracy = model.evaluate(testX, testy, batch_size=10, verbose=1)
    print('Accuracy :', accuracy)


evaluate_emergency_vehicles()
