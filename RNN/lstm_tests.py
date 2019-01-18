from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
from keras.utils import to_categorical
import numpy as np

n_examples = 200
n_features = 1

class_dict = {
    0: "miou",
    1: 'pattes',
    2: "greu"
}

model = Sequential()

model.add(LSTM(32, return_sequences=True, input_shape=(None, n_features)))
model.add(LSTM(8, return_sequences=True))
model.add(TimeDistributed(Dense(3, activation='sigmoid')))

print(model.summary(90))

model.compile(loss='categorical_crossentropy', optimizer='adam')


def train_generator():
    while True:
        sequence_length = np.random.randint(4, 10)

        desired_class = np.random.randint(0, 3)

        x_train = np.array([np.transpose([[desired_class for i in range(sequence_length)]])])
        y_train = [[to_categorical(desired_class, 3) for i in range(sequence_length)]]
        desired_class = np.random.randint(0, 3)

        for example in range(n_examples - 1):
            x_train = np.concatenate((x_train, [np.transpose([[desired_class for i in range(sequence_length)]])]))
            y_train = np.concatenate((y_train, [[to_categorical(desired_class, 3) for i in range(sequence_length)]]))
            desired_class = np.random.randint(0, 3)
        yield x_train, y_train


# for x_train, y_train in train_generator():
#     print(x_train.shape)
#     print(x_train)
#     print(y_train.shape)
#     print(y_train)
#     exit()

model.fit_generator(train_generator(), steps_per_epoch=30, epochs=1, verbose=2)


x = np.array([np.transpose([[2 for i in range(11)]])])

print(x.shape)

print(model.predict(x, batch_size=1, verbose=0, steps=None))
