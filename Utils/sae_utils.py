import numpy as np

from keras import Sequential, Input, Model
from keras.layers import Dense
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid, KFold


def create_sae(input_dim, structure, optimizer):
    sae_input = Input(shape=(input_dim,))

    # Encoder part
    encoded = Dense(structure[0], activation='relu')(sae_input)
    if len(structure) > 1:
        for i in range(1, len(structure)):
            encoded = Dense(structure[i])(encoded)

    # Decoder part
    decoded = Dense(structure[-2], activation='relu')(encoded)
    if len(structure) > 2:
        for i in range(2, len(structure)):
            decoded = Dense(structure[len(structure) - (i + 1)], activation='relu')(decoded)

    decoded = Dense(input_dim, activation='relu')(decoded)

    autoencoder = Model(sae_input, decoded)
    autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')

    encoder = Model(sae_input, encoded)

    return autoencoder, encoder


def optimize_sae(x, cv):
    optimizer = 'adam'
    batch_size = [10, 50]
    epochs = [10, 50, 100]
    structure = [
        # 3 layers auto encoders
        [40, 20, 10], [40, 20, 5], [40, 15, 10], [40, 15, 5],
        [35, 20, 10], [35, 20, 5], [35, 15, 10], [35, 15, 5],
        [30, 20, 10], [30, 20, 5], [30, 15, 10], [30, 15, 5],
        [25, 20, 10], [25, 20, 5], [25, 15, 10], [25, 15, 5],

        # 2 layers auto encoders
        [40, 20], [40, 15], [40, 10],
        [35, 20], [35, 15], [35, 10],
        [30, 20], [30, 15], [30, 10],
        [25, 20], [25, 15], [25, 10]
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
        for train, test in kf.split(x):
            print('\tIteration', i)

            train_x = x[train]
            validation_x = x[test]

            # initialize the entry of the dict

            # Create the model
            model, _ = create_sae(train_x.shape[1], combination['structure'], optimizer=optimizer)

            # Train the model
            model.fit(train_x, train_x, batch_size=combination['batch_size'], epochs=combination['epochs'], verbose=0)

            # Evaluate the model
            test_loss = model.evaluate(validation_x, validation_x, verbose=0)

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


def optimize_mlp(x, y, cv):
    # Create the hyper parameters possible values
    batch_size = [20, 40]
    epochs = [50, 100]
    structure = [
        None,
        [5]
    ]

    param_grid_mlp = dict(
        batch_size=batch_size,
        epochs=epochs,
        structure=structure
    )

    possible_combination = list(ParameterGrid(param_grid_mlp))
    search_results = dict()

    for combination in possible_combination:
        print('Testing combination', str(combination))

        search_results[str(combination)] = []

        kf = KFold(n_splits=cv)

        # Cross validate results
        i = 1
        for train, test in kf.split(x):
            print('\tIteration', i)

            train_x = x[train]
            validation_x = x[test]

            train_y = y[train]
            validation_y = y[test]

            # Create the model
            model = create_mlp(
                input_dim=x.shape[1],
                n_classes=y.shape[1],
                optimizer='adam',
                structure=combination['structure']
            )

            # Train the model
            model.fit(train_x, train_y, batch_size=combination['batch_size'], epochs=combination['epochs'], verbose=0)

            # Make predictions
            predictions = model.predict_classes(validation_x)

            # Assess and store performances
            search_results[str(combination)].append(
                np.round(
                    f1_score(y_true=np.argmax(validation_y, axis=1), y_pred=predictions, average='weighted'),
                    decimals=7
                )
            )

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
