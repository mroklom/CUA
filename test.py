import numpy as np
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras_preprocessing.sequence import pad_sequences


def generate_class(x):
    last = x[-1]
    total = 0
    for point in last:
        total += point

    if total > 15:
        return [1, 0]
    else:
        return [0, 1]


def get_max_length_sequence(data):
    max_length = 0
    for i in range(len(data)):
        if data[i].shape[0] > max_length:
            max_length = data[i].shape[0]

    return max_length


def preprocess_sequences(data, max_length):
    new_data = []

    for sequence in data:

        new_sequence = np.array(sequence[0:sequence.shape[0]])

        diff_length = max_length - sequence.shape[0]
        if diff_length != 0:
            for i in range(diff_length):
                new_sequence = np.concatenate((new_sequence, np.array([[0, 0, 0]])))

        new_data.append(new_sequence)

    return np.array(new_data)


n_train = 10000
n_test = 300
n_features = 3

trainX = np.array([
    np.array([
        np.array([
            np.random.randint(0, 10) for i in range(n_features)
        ]) for j in range(np.random.randint(2, 5))
    ]) for k in range(n_train)
])

trainy = np.array([generate_class(trainX[i]) for i in range(n_train)])

testX = np.array([
    np.array([
        np.array([
            np.random.randint(0, 10) for i in range(n_features)
        ]) for j in range(np.random.randint(2, 5))
    ]) for k in range(n_test)
])

testY = np.array([generate_class(testX[i]) for i in range(n_test)])

trainX = preprocess_sequences(trainX, get_max_length_sequence(trainX))
testX = preprocess_sequences(testX, get_max_length_sequence(testX))

model = Sequential()
model.add(LSTM(5, input_shape=(None, n_features)))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x=trainX, y=trainy, batch_size=100, epochs=10, verbose=2)
loss, accuracy = model.evaluate(x=testX, y=testY, batch_size=1, verbose=2)
print(accuracy)

predX = [
    np.array([
        [1, 2, 2],
        [4, 5, 6],
        [1, 1, 1]
    ]),
    np.array([
        [1, 2, 2],
        [9, 9, 9]
    ])
]

for to_pred in predX:
    todo = np.array([to_pred])
    predictions = model.predict(todo, batch_size=1, verbose=2)
    print(predictions)
