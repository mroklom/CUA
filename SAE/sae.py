import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# read data
mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
X_train = np.array([np.reshape(matrix, 784) for matrix in train_images])
X_test = np.array([np.reshape(matrix, 784) for matrix in test_images])

# Convert class to one hot encoding vector
train_labels_temp = np.zeros((len(train_labels), 10))
test_labels_temp = np.zeros((len(test_labels), 10))
train_labels_temp[np.arange(len(train_labels)), train_labels] = 1
test_labels_temp[np.arange(len(test_labels)), test_labels] = 1
train_labels = train_labels_temp
test_labels = test_labels_temp

# declare some constants
num_input = 784
num_hidden1 = 392
num_hidden2 = 196
num_hidden3 = num_hidden1
num_outputs = num_input
learning_rate_encoder = 0.01
learning_rate_mlp = 0.001
activation_function = tf.nn.relu
num_classes = 10

# Input
X_encoder = tf.placeholder(tf.float32, shape=[None, num_input])
X = tf.placeholder(tf.float32, shape=[None, num_hidden2])
Y = tf.placeholder(tf.float32, shape=[None, num_classes])

initializer = tf.variance_scaling_initializer()

# Weights
weights = {
    "encoder": {
        "hidden1": tf.Variable(initializer([num_input, num_hidden1]), dtype=tf.float32),
        "hidden2": tf.Variable(initializer([num_hidden1, num_hidden2]), dtype=tf.float32),
        "hidden3": tf.Variable(initializer([num_hidden2, num_hidden3]), dtype=tf.float32),
        "output": tf.Variable(initializer([num_hidden3, num_outputs]), dtype=tf.float32),
    },
    "mlp": {
        "hidden1": tf.Variable(initializer([num_hidden2, 60]), dtype=tf.float32),
        "hidden2": tf.Variable(initializer([60, 30]), dtype=tf.float32),
        "class": tf.Variable(initializer([30, num_classes]), dtype=tf.float32)
    }
}

biases = {
    "encoder": {
        "hidden1": tf.Variable(tf.zeros(num_hidden1)),
        "hidden2": tf.Variable(tf.zeros(num_hidden2)),
        "hidden3": tf.Variable(tf.zeros(num_hidden3)),
        "output": tf.Variable(tf.zeros(num_outputs))
    },
    "mlp": {
        "hidden1": tf.Variable(tf.zeros(60)),
        "hidden2": tf.Variable(tf.zeros(30)),
        "class": tf.Variable(tf.zeros(num_classes))
    }
}
# Biases

# Layers
hidden_layer1 = activation_function(
    tf.matmul(X_encoder, weights["encoder"]["hidden1"]) + biases["encoder"]["hidden1"]
)

hidden_layer2 = activation_function(
    tf.matmul(hidden_layer1, weights["encoder"]["hidden2"]) + biases["encoder"]["hidden2"]
)

hidden_layer3 = activation_function(
    tf.matmul(hidden_layer2, weights["encoder"]["hidden3"]) + biases["encoder"]["hidden3"]
)

output_layer = activation_function(
    tf.matmul(hidden_layer3, weights["encoder"]["output"]) + biases["encoder"]["output"]
)

mlp_hidden1_layer = activation_function(
    tf.matmul(X, weights["mlp"]["hidden1"]) + biases["mlp"]["hidden1"]
)

mlp_hidden2_layer = activation_function(
    tf.matmul(mlp_hidden1_layer, weights["mlp"]["hidden2"]) + biases["mlp"]["hidden2"]
)

class_layer = tf.matmul(mlp_hidden2_layer, weights["mlp"]["class"]) + biases["mlp"]["class"]

# Training metric
loss_encoder = tf.reduce_mean(tf.square(output_layer - X_encoder))
loss_mlp = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=class_layer, labels=Y))

# Optimizer
optimizer_encoder = tf.train.AdamOptimizer(learning_rate_encoder)
train_encoder = optimizer_encoder.minimize(loss_encoder)

optimizer_mlp = tf.train.AdamOptimizer(learning_rate=learning_rate_mlp)
train_mlp = optimizer_mlp.minimize(loss_mlp)

init = tf.global_variables_initializer()

# Training parameters
num_epoch_encoder = 15
num_epoch_mlp = 15

batch_size = 150
num_batches = len(train_labels) // batch_size  # 400
num_test_images = 10

# Run training
with tf.Session() as sess:
    sess.run(init)

    # Train the encoder
    print("---------- TRAINING ENCODER PART ----------")
    for epoch in range(num_epoch_encoder):

        print("Running epoch", epoch)

        for batch in range(num_batches):
            begin_index = batch * batch_size
            end_index = (batch + 1) * batch_size
            X_batch = X_train[begin_index:end_index]

            sess.run(train_encoder, feed_dict={X_encoder: X_batch})

        train_loss = loss_encoder.eval(feed_dict={X_encoder: X_batch})
        print("\tloss = {}\n".format(train_loss))

    # Evaluate on the test images
    results = output_layer.eval(feed_dict={X_encoder: X_test})

    # Plot the results of the reconstruction
    f, a = plt.subplots(2, num_test_images)
    for i in range(num_test_images):
        a[0][i].imshow(test_images[i])
        a[1][i].imshow(np.reshape(results[i], (28, 28)))
    plt.show()

    print("\n")

    # Train the MLP
    print("------------ TRAINING MLP PART ------------")
    for epoch in range(num_epoch_mlp):
        print("Running epoch", epoch)
        avg_cost = 0

        for batch in range(num_batches):
            begin_index = batch * batch_size
            end_index = (batch + 1) * batch_size
            X_batch = X_train[begin_index:end_index]
            Y_batch = train_labels[begin_index:end_index]

            _, c = sess.run([train_mlp, loss_mlp], feed_dict={
                X: hidden_layer2.eval(feed_dict={X_encoder: X_batch}),
                Y: Y_batch
            })

            avg_cost += c / num_batches

        print("\tloss = {}\n".format(avg_cost))

    print("Optimization finished!")

    # Test model
    pred = tf.nn.softmax(class_layer)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))

    # Accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({
        X: hidden_layer2.eval(feed_dict={X_encoder: X_test}),
        Y: test_labels
    }))
