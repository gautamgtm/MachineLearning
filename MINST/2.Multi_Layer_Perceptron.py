#-----------------------------------------Import Libraries---------------------------------------#
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow_models import ConvolutionalNN, FullyConnectedNN

#-----------------------------------------Model Parameters----------------------------------------#
_INPUT_UNITS = 784
_HIDDEN_UNITS = 64
_OUTPUT_UNITS = 10
_LEARNING_RATE = 0.5
_NEPOCH = 3000
_BATCH_SIZE = 50

#-----------------------------------------Import Data---------------------------------------#
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#-----------------------------------------Create Model---------------------------------------#
print("Creating a Neural Network.")

### Define Inputs and Outputs ###
x = tf.placeholder(tf.float32, [None, _INPUT_UNITS])
y_ = tf.placeholder(tf.float32, [None, _OUTPUT_UNITS])

#h_fc2, keep_prob = FullyConnectedNN(x,_INPUT_UNITS,_HIDDEN_UNITS,_OUTPUT_UNITS)
### Define Weights and Biases associated with each layer ###
norm = np.sqrt(_INPUT_UNITS+_HIDDEN_UNITS)
W1 = tf.Variable(tf.random_uniform([_INPUT_UNITS, _HIDDEN_UNITS], minval=-1/norm, maxval=1/norm, seed = 123), name='W1')
b1 = tf.Variable(tf.random_uniform([_HIDDEN_UNITS], minval=-1/np.sqrt(_INPUT_UNITS), maxval=1/np.sqrt(_INPUT_UNITS)), name='b1')

norm = np.sqrt(_HIDDEN_UNITS+_OUTPUT_UNITS)
W2 = tf.Variable(tf.random_uniform([_HIDDEN_UNITS, _OUTPUT_UNITS], minval=-1/norm, maxval=1/norm), name='W2')
b2 = tf.Variable(tf.random_uniform([_OUTPUT_UNITS], minval=-1/norm, maxval=1/norm), name='b2')

### Define the Operations throughout the NN ###
h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
y = tf.matmul(h1, W2) + b2

#-----------------------------------------Define Loss and Optimizer---------------------------------------#
print("Compiling the Neural Network.")
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(_LEARNING_RATE).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1), tf.argmax(y_,1)), tf.float32), name='acc')

#-----------------------------------------Create TF Session---------------------------------------#
print("Creating the Tensorflow Session.")
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#-----------------------------------------Training Model---------------------------------------#
print("Training the Neural Network : Started.")
for _ in range(_NEPOCH):
    batch_xs, batch_ys = mnist.train.next_batch(_BATCH_SIZE)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
print("Training the Neural Network : Completed.")

#-----------------------------------------Testing Model---------------------------------------#
print("Testing the Neural Network.")
print("Testing Accuracy: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

#-----------------------------------------End of Code!---------------------------------------#'''
