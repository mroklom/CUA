import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import ParameterGrid, KFold
from sklearn.metrics import f1_score


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
    batch_size = [10, 50]
    epochs = [10, 50, 100]
    n_units_output_lstm = [100, 90, 80, 70]
    n_units_output_dense = [40, 30, 20]

    # Create the hyper parameters grid
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

            train_x = x[train]
            validation_x = x[test]

            train_y = y[train]
            validation_y = y[test]

            # Create the model
            model = create_model(
                n_units_output_lstm=combination['n_units_output_lstm'],
                n_units_output_dense=combination['n_units_output_dense'],
                n_timesteps=n_timesteps,
                n_features=n_features,
                n_classes=n_classes
            )

            # Train the model
            model.fit(train_x, train_y, batch_size=combination['batch_size'], epochs=combination['epochs'], verbose=2)

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
