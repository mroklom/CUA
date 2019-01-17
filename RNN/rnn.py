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
from loader import load_data, return_observation_by_observation
import os
import numpy as np

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# load a single file as a numpy array
def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    return loaded


# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_' + group + '.txt', 'total_acc_y_' + group + '.txt', 'total_acc_z_' + group + '.txt']
    # body acceleration
    filenames += ['body_acc_x_' + group + '.txt', 'body_acc_y_' + group + '.txt', 'body_acc_z_' + group + '.txt']
    # body gyroscope
    filenames += ['body_gyro_x_' + group + '.txt', 'body_gyro_y_' + group + '.txt', 'body_gyro_z_' + group + '.txt']
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/y_' + group + '.txt')
    return X, y


# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
    # load all train
    trainX, trainy = load_dataset_group('train', prefix)
    # print(trainX.shape, trainy.shape)
    # load all test
    testX, testy = load_dataset_group('test', prefix)
    # print(testX.shape, testy.shape)
    # zero-offset class values
    trainy = trainy - 1
    testy = testy - 1
    # one hot encode y
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    # print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX, trainy, testX, testy


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
    trainX, trainy, testX, testy = load_dataset('/users/21509823t/PycharmProjects/CUA/Data/UCI HAR Dataset/')
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
    epochs = [10, 50]
    n_units_output_lstm = [100, 150]
    n_units_output_dense = [100, 150]

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
        cv=3,
        verbose=2
    )

    grid_result = grid.fit(trainX, trainy, verbose=2)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


# Fit and evaluate the model
def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 2, 15, 64
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit the model
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)

    # evaluate the model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=verbose)
    return accuracy


# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


# Run multiple experiments
def run_experiment(repeats=10):
    trainX, trainy, testX, testy = load_dataset('/home/cua/Documents/UCI HAR Dataset/')
    scores = list()
    for r in range(repeats):
        print('Iteration ', r + 1)
        score = evaluate_model(trainX, trainy, testX, testy)
        score = score * 100.0
        print('>#%d: %.3f' % (r + 1, score))
        scores.append(score)
        print()
    summarize_results(scores)


def evaluate_emergency_vehicles():
    trainX, trainy = load_data('/users/21509823t/PycharmProjects/CUA/Data/export-debut.csv')
    trainX2, trainy2, testX2, testy2 = load_dataset('/users/21509823t/PycharmProjects/CUA/Data/UCI HAR Dataset/')

    print(trainX[0].shape)
    print(trainX2[0].shape)

    print()

    print(trainX.shape)
    print(trainX2.shape)

    print(trainX)
    print(trainX2)

    exit()

    n_batch = 1
    n_epoch = 1

    model = Sequential()
    model.add(LSTM(50, batch_input_shape=(1, None, 9)))
    model.add(Dropout(0.5))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(6, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    data_path = '/users/21509823t/PycharmProjects/CUA/Data/export-debut.csv'

    model.fit(trainX2, trainy2, epochs=n_epoch, batch_size=n_batch, verbose=1, shuffle=False)


prefix = '/users/21509823t/PycharmProjects/CUA/Data/UCI HAR Dataset/'
# run_experiment()
# optimize_model()

evaluate_emergency_vehicles()
