import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist

# load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Training set: 60,000 samples; each sample is a 28x28 grayscale image
# Test set:     10,000 samples; each sample is a 28x28 grayscale image
print('X_train.shape:', X_train.shape)  # X_train.shape: (60000, 28, 28)
print('X_test.shape:', X_test.shape)    # X_test.shape: (10000, 28, 28)

# flatten the images in X_train and X_test such that each sample becomes a row vector
X_train = np.array([sample.flatten() for sample in X_train])
X_test = np.array([sample.flatten() for sample in X_test])

print('X_train.shape:', X_train.shape)  # X_train.shape: (60000, 784)
print('X_test.shape:', X_test.shape)    # X_test.shape: (10000, 784)

# convert the pixel values from integer to float32 and 
# normalize the pixel values from the range of 0-255 to 0-1
X_train.astype('float32')
X_train = X_train / 255
X_test.astype('float32')
X_test = X_test / 255

print('X_train range: ', np.min(X_train), ', ', np.max(X_train))
print('X_test range: ', np.min(X_test), ', ', np.max(X_test))

# determine number of pixels in an input image
num_pixels = X_train.shape[1]

# determine number of classes
num_classes = 10

# define a deep neural network model using the sequential model API
# Layer 0: input layer specifying the dimension of each sample
# Layer 1: n^[1] = 800 nodes, g^[1] = ReLU
# Layer 2: n^[2] = 100 nodes, g^[2] = ReLU
# Layer 3: n^[3] = num_classes nodes, g^[3] = softmax
model = Sequential()
model.add(Input(shape=(784,)))
model.add(Dense(800, activation=tf.nn.relu))
model.add(Dense(100, activation=tf.nn.relu))
model.add(Dense(10, activation=tf.nn.softmax))

# print a summary of the model
model.summary()

# compile the model using
# a. Optimizer: gradient descent SGD with a learning rate of 0.1
# b. Loss function: sparse_categorical_crossentropy 
# c. Metrics: accuracy
model.compile(optimizer=SGD(learning_rate=0.1),
              loss='sparse_categorical_crossentropy',
              metrics='accuracy')
 
# fit the model to training data
model.fit(X_train, y_train, epochs = 10, verbose = 1)
 
# evaluate the model on the test data
loss, acc = model.evaluate(X_test, y_test, verbose = 1)
print('Test accuracy = %.3f' % acc)
