import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import os

from sklearn.model_selection import GridSearchCV

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_data(with_labels=True):
    # load data
    mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    x_train = np.array([np.reshape(matrix / 255.0, 784) for matrix in train_images])
    x_test = np.array([np.reshape(matrix / 255.0, 784) for matrix in test_images])

    y_train = to_categorical(train_labels)
    y_test = to_categorical(test_labels)

    if with_labels:
        return x_train, y_train, x_test, y_test
    else:
        return x_train, x_test


def create_autoencoder_1_layers(n_neurones_1):
    model = Sequential()
    model.add(Dense(n_neurones_1, activation='relu', input_dim=784))
    model.add(Dense(784, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model


def optimize():
    # Load  the data
    train_x, test_x = load_data(with_labels=False)
    train_x_reshape = np.array([train_x[i].reshape(784, 1) for i in range(train_x.shape[0])])

    # Map the model to scikit learn
    model = KerasClassifier(build_fn=create_autoencoder_1_layers, verbose=0)

    param_grid = dict(
        batch_size=[150],
        epochs=[1],
        n_neurones_1=[400, 300],
        # n_neurones_2=[200, 100]
    )

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, verbose=1,
                        scoring='neg_mean_squared_error')

    grid_result = grid.fit(X=train_x, y=train_x_reshape, verbose=2)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


optimize()

# x_train, x_test = load_data(with_labels=False)
# model = create_autoencoder_1_layers(n_neurones_1=400)
# model.fit(x=x_train, y=x_train, batch_size=150, epochs=1, validation_data=(x_test, x_test), verbose=2)
