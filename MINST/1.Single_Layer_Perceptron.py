#-----------------------------------------Import Libraries---------------------------------------#
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

#-----------------------------------------Model Parameters----------------------------------------#
_INPUT_UNITS = 784
_OUTPUT_UNITS = 10
_LEARNING_RATE = 0.5
_NEPOCH = 1000
_BATCH_SIZE = 100

#-----------------------------------------Import Data---------------------------------------#
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#-----------------------------------------Create Model---------------------------------------#
print("Creating a Neural Network.")
x = tf.placeholder(tf.float32, [None, _INPUT_UNITS])

norm = np.sqrt(_INPUT_UNITS+_OUTPUT_UNITS)
W = tf.Variable(tf.random_uniform([_INPUT_UNITS, _OUTPUT_UNITS], minval=-1/norm, maxval=1/norm, seed = 123))
b = tf.Variable(tf.random_uniform([_OUTPUT_UNITS], minval=-1/norm, maxval=1/norm))
y = tf.matmul(x, W) + b

#-----------------------------------------Define Loss and Optimizer---------------------------------------#
print("Compiling the Neural Network.")
y_ = tf.placeholder(tf.float32, [None, _OUTPUT_UNITS])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(_LEARNING_RATE).minimize(cross_entropy)

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
print("Testing the Neural Network : Started.")
correct_predictions = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
print("Testing the Neural Network : Completed.")
print("Testing Accuracy: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

#-----------------------------------------End of Code!---------------------------------------#'''
