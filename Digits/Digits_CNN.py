import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist

# load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Training set: 60,000 samples; each sample is a 28x28 grayscale image
# Test set:     10,000 samples; each sample is a 28x28 grayscale image
print('X_train.shape:', X_train.shape)
print('X_test.shape:', X_test.shape)

# reshape into a 4-D array
X_train = X_train.reshape([X_train.shape[0], X_train.shape[1], X_train.shape[2], 1])
X_test = X_test.reshape([X_test.shape[0], X_test.shape[1], X_test.shape[2], 1])

print('X_train.shape:', X_train.shape)
print('X_test.shape:', X_test.shape)

# convert the pixel values from integer to float32 and 
# normalize the pixel values from the range of 0-255 to 0-1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print('X_train range: ', np.min(X_train), ', ', np.max(X_train))
print('X_test range: ', np.min(X_test), ', ', np.max(X_test))

# determine number of pixels in an input image
num_pixels = X_train.shape[1]

# determine number of classes
num_classes = 10

# define a deep neural network model using the sequential model API
# Layer 0: input layer specifying the dimension of each sample
# Layer 1: 2D convolution layer with 32 filters, each filter of dimension 3x3, using ReLU activation function
# Layer 2: 2D max pooling layer with filter dimension of 2 x 2
# Layer 3: Flatten the images into a column vector
# Layer 4: Fully connected NN layer with n = 100 nodes, g = ReLU
# Layer 5: Fully connected NN layer with n = num_classes nodes, g = softmax
# -------- ENTER YOUR CODE HERE --------
cnn_model = Sequential()
cnn_model.add(Input(shape=(num_pixels, num_pixels, 1)))
cnn_model.add(Conv2D(32, (3, 3), activation='relu'))
cnn_model.add(MaxPooling2D(2,2))
cnn_model.add(Flatten())
cnn_model.add(Dense(100, activation='relu'))
cnn_model.add(Dense(num_classes, activation='softmax'))
# -------- END YOUR CODE HERE --------

# print a summary of the model
cnn_model.summary()

# compile the model using
# a. Optimizer: gradient descent  with a learning rate of 0.1
# b. Loss function: sparse_categorical_crossentropy 
# c. Metrics: accuracy
opt = SGD(learning_rate=0.1)
cnn_model.compile(optimizer = opt, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
 
# fit the model to training data
cnn_model.fit(X_train, y_train, epochs = 10, verbose = 1)
 
# evaluate the model on the test data
loss, acc = cnn_model.evaluate(X_test, y_test, verbose = 1)
print('Test accuracy = %.3f' % acc)
