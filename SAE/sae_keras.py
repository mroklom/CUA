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

# Hyper Parameters
# SAE
encoded_dims = [300, 100, 25]
sae_epoch = 25
sae_batch_size = 150

# MLP
mlp_epoch = 100
mlp_batch_size = 150

# Declare and compile the models
# SAE
input_img = Input(shape=(784,))
encoded = Dense(encoded_dims[0], activation='relu')(input_img)
encoded = Dense(encoded_dims[1], activation='relu')(input_img)
encoded = Dense(encoded_dims[2], activation='relu')(input_img)

decoded = Dense(encoded_dims[1], activation="relu")(encoded)
decoded = Dense(encoded_dims[0], activation="relu")(encoded)
decoded = Dense(784, activation="sigmoid")(encoded)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoded_dims[-1],))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# MLP
mlp_layer = Dense(15, activation='relu')(encoded_input)
mlp_layer = Dense(10, activation='softmax')(mlp_layer)
mlp = Model(encoded_input, mlp_layer)
mlp.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

# Fit the models
# SAE
autoencoder.fit(
    x=X_train,
    y=X_train,
    epochs=sae_epoch,
    batch_size=sae_batch_size,
    shuffle=True,
    validation_data=(X_test, X_test),
    verbose=2
)

# Compute the encoded version of the data
encoded_images_train = encoder.predict(X_train)
encoded_images_test = encoder.predict(X_test)
decoded_images = decoder.predict(encoded_images_test)

# MLP
mlp.fit(
    x=encoded_images_train,
    y=y_train,
    batch_size=mlp_batch_size,
    epochs=mlp_epoch,
    verbose=2,
    validation_data=(encoded_images_test, y_test)
)

# Print performances
score = mlp.evaluate(encoded_images_test, y_test, verbose=0)
print('Test accuracy:', score[1])

# Print some reconstructed images
n = 10
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
    plt.imshow(decoded_images[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
