from keras import Sequential, Input, Model
from keras.layers import Dense
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, ParameterGrid, train_test_split, KFold

from loader import load_data_sae
import pandas as pd

import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def create_sae(input_dim, structure, optimizer):
    input = Input(shape=(input_dim,))

    # Encoder part
    encoded = Dense(structure[0], activation='relu')(input)
    if len(structure) > 1:
        for i in range(1, len(structure)):
            encoded = Dense(structure[i])(encoded)

    # Decoder part
    decoded = Dense(structure[-2], activation='relu')(encoded)
    if len(structure) > 2:
        for i in range(2, len(structure)):
            decoded = Dense(structure[len(structure) - (i + 1)], activation='relu')(decoded)

    decoded = Dense(input_dim, activation='relu')(decoded)

    autoencoder = Model(input, decoded)
    autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')

    encoder = Model(input, encoded)

    return autoencoder, encoder


def optimize_sae(X, cv):
    optimizer = 'adam'
    batch_size = [20, 40]
    epochs = [50, 100]
    structure = [
        [20, 10],
        [30, 20, 10]
    ]

    param_grid = dict(
        batch_size=batch_size,
        epochs=epochs,
        structure=structure
    )

    possible_combination = list(ParameterGrid(param_grid))
    search_results = dict()

    for combination in possible_combination:

        print('Testing combination', str(combination))

        search_results[str(combination)] = []

        kf = KFold(n_splits=cv)

        # Cross validate results
        i = 1
        for train, test in kf.split(X):
            print('\tIteration', i)

            trainX = X[train]
            validationX = X[test]

            # initialize the entry of the dict

            # Create the model
            model, _ = create_sae(trainX.shape[1], combination['structure'], optimizer=optimizer)

            # Train the model
            model.fit(trainX, trainX, batch_size=combination['batch_size'], epochs=combination['epochs'], verbose=0)

            # Evaluate the model
            test_loss = model.evaluate(validationX, validationX, verbose=0)

            # Store the results in the dict
            search_results[str(combination)].append(test_loss)

            # increment iteration counter
            i = i + 1

        print('Results for', str(combination))
        print(search_results[str(combination)])
        print()

    # Find the best combination of parameters
    best_params_list = []
    min_loss = float("inf")
    for combination, losses in search_results.items():
        mean_loss = np.mean(losses)
        if mean_loss <= min_loss:
            if mean_loss < min_loss:
                best_params_list = [combination]
                min_loss = mean_loss
            elif mean_loss == min_loss:
                best_params_list.append(combination)

    return best_params_list, min_loss, search_results


def create_mlp(input_dim, n_classes, optimizer, structure=None):
    model = Sequential()

    if structure is not None:
        model.add(Dense(structure[0], activation='relu', input_dim=input_dim))
        if len(structure) > 1:
            for i in range(1, len(structure)):
                model.add(Dense(structure[i], activation='relu'))

        model.add(Dense(n_classes, activation='softmax'))

    else:
        model.add(Dense(n_classes, activation='relu', input_dim=input_dim))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def optimize_mlp(X, y):
    optimizer = 'adam'
    batch_size = [20, 40]
    epochs = [50, 100]
    structure = [
        None,
        [5]
    ]

    model_mlp = KerasClassifier(
        build_fn=create_mlp,
        input_dim=X.shape[1],
        n_classes=y.shape[1],
        optimizer=optimizer,
        verbose=0
    )

    param_grid_mlp = dict(
        batch_size=batch_size,
        epochs=epochs,
        structure=structure
    )

    grid_mlp = GridSearchCV(
        estimator=model_mlp,
        param_grid=param_grid_mlp,
        n_jobs=-1,
        cv=3,
        verbose=1
    )

    grid_result_mlp = grid_mlp.fit(X, y, verbose=0)

    print("Best: %f using %s" % (grid_result_mlp.best_score_, grid_result_mlp.best_params_))

    means = grid_result_mlp.cv_results_['mean_test_score']
    stds = grid_result_mlp.cv_results_['std_test_score']
    params = grid_result_mlp.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    return grid_result_mlp.best_params_


data = pd.DataFrame(load_data_sae('/home/cua/PycharmProjects/CUA/Data/export-debut.csv'))

X = data.drop(columns=['class'], axis=1).values
y = to_categorical(data['class'].values - 1, np.unique(data['class'].values).shape[0])

trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.25)

# Optimize the sae
print('Optimizing SAE...\n')
params, loss, results = optimize_sae(trainX, cv=3)
best_params_sae = eval(params[0])
print('\nOptimizing SAE done!\n\n')

# Create the sae with it's best parameters and train it
print('Training SAE...\n')
sae, e = create_sae(input_dim=trainX.shape[1], structure=best_params_sae['structure'], optimizer='adam')
sae.fit(trainX, trainX, batch_size=best_params_sae['batch_size'], epochs=best_params_sae['epochs'], verbose=2)
print('Training SAE done!\n\n')

# Encode train X
encodedTrainX = e.predict(trainX)

# Optimize the mlp
print('Optimizing MLP...\n')
best_params_mlp = optimize_mlp(encodedTrainX, trainy)
print('Optimizing MLP done!\n\n')

# Create the sae with it's best parameters and train it
print('Training MLP...\n')
mlp = create_mlp(input_dim=encodedTrainX.shape[1], n_classes=trainy.shape[1], optimizer='adam', structure=best_params_mlp['structure'])
mlp.fit(encodedTrainX, trainy, batch_size=best_params_mlp['batch_size'], epochs=best_params_mlp['epochs'], verbose=2)
print('Training MLP done!\n\n')

# Evaluate the model
encodedTestX = e.predict(testX)
loss_mlp, accuracy_mlp = mlp.evaluate(encodedTestX, testy, verbose=0)
print('Accuracy on test set:', accuracy_mlp)
