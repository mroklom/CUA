import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Input
import os

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# load data
mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
X_train = np.array([np.reshape(matrix / 255.0, 784) for matrix in train_images])
X_test = np.array([np.reshape(matrix / 255.0, 784) for matrix in test_images])

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

encoded_dim = 100

input_img = Input(shape=(784,))
encoded = Dense(encoded_dim, activation='relu')(input_img)
decoded = Dense(784, activation="sigmoid")(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded input
encoded_input = Input(shape=(encoded_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='mean_squared_error')

autoencoder.fit(X_train, X_train,
                epochs=5,
                batch_size=150,
                shuffle=True,
                validation_data=(X_test, X_test),
                verbose=2)

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(X_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# model = Sequential()
# model.add(Dense(784, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(784, activation='relu'))
#
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
#
# model.fit(X_train, X_train, epochs=5, batch_size=150, verbose=2)
#
# _, perf = model.evaluate(X_test, X_test, batch_size=150, verbose=2)
# print("performances on test set : ", perf)
#
# # show the reconstruction of the image
# predictions = model.predict(X_test[:5])
# f, a = plt.subplots(2, 5)
# for i, prediction in enumerate(predictions):
#     reconstructed_image = np.reshape(prediction, (28, 28))
#     base_image = np.reshape(X_test[i], (28, 28))
#     a[0][i].imshow(base_image)
#     a[1][i].imshow(reconstructed_image)
# plt.show()
