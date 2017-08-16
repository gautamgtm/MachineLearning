#-----------------------------------------Import Libraries---------------------------------------#
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import init_ops
from tensorflow_models import ConvolutionalNN, FullyConnectedNN, RecurrentNN, freeze_graph, load_graph

#-----------------------------------------Model Parameters----------------------------------------#
_input_units = 28
_time_steps = 28
_hidden_units = 256
_output_units = 10
_learning_rate = 1
_nepoch = 50000
_batch_size = 128
_patience = 5000

#-----------------------------------------Import Data---------------------------------------#
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#-----------------------------------------Create Model---------------------------------------#
print("Creating a Neural Network.")

# Define Inputs and Outputs
x = tf.placeholder(tf.float32, [None, _input_units, _time_steps], name="input")
y_ = tf.placeholder(tf.float32, [None, _output_units], name="output")

# Define Neural Network Layers
# A Simple RNN Layer

rnn_cell = tf.nn.rnn_cell.BasicRNNCell(_hidden_units, activation=tf.nn.relu)
w_r1 = tf.Variable(tf.random_normal([_hidden_units, _output_units]), name='w_r1')
b_r1 = tf.Variable(tf.random_normal([_output_units]), name='b_r1')

o_r1, s_r1 = tf.nn.static_rnn(rnn_cell, tf.unstack(x, _time_steps, 1), #Unstack to get a list of 'time steps' tensors of shape (batch sz, input unit)
                dtype=tf.float32)
h_r1 = tf.matmul(o_r1[-1], w_r1) + b_r1

#h_r1 = RecurrentNN(x,_input_units,_hidden_units,_output_units,_time_steps)

#-----------------------------------------Define Loss and Optimizer---------------------------------------#
print("Compiling the Neural Network.")
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=h_r1), name="loss")
train_step = tf.train.AdadeltaOptimizer(_learning_rate).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(h_r1,1), tf.argmax(y_,1)), tf.float32), name="accuracy")

#-----------------------------------------Create TF Session---------------------------------------#
print("Creating the Tensorflow Session.")
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
saver = tf.train.Saver()

#-----------------------------------------Training Model---------------------------------------#
print("Training the Neural Network : Started.")
max_val_accuracy, cnt = 0, 0
for i in range(_nepoch):
    batch_x, batch_y = mnist.train.next_batch(_batch_size)
    # Reshape data to get 28 seq of 28 elements
    batch_x = batch_x.reshape((-1, _time_steps, _input_units))
    sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
    # Monitor Validation Loss (Every 10 epchs) to save the best model and early stopping
    cnt = cnt + 1
    if cnt == _patience:
        break

    if i%50 == 0:
        val_x, val_y = mnist.validation.images.reshape((-1, _time_steps, _input_units)), mnist.validation.labels
        val_accuracy = accuracy.eval(feed_dict={x: val_x, y_: val_y})
        train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: batch_y})
        print('Step %d: Train Accuracy: %g Validation Accuracy: %g' % (i, train_accuracy, val_accuracy))
        if val_accuracy > max_val_accuracy:
            print('Step %d: Better Validation Accuracy Achieved. Saved the model.' % (i))
            saver.save(sess, './models/mnist')
            max_val_accuracy = val_accuracy
            cnt = 0

print("Training the Neural Network : Completed.")

#-----------------------------------------Freeze Model---------------------------------------#
print("Freezing the best model. Saving the Frozen Model.")
freeze_graph("mnist")

#-----------------------------------------Testing Model---------------------------------------#
print("Testing the Neural Network.")
test_x, test_y = mnist.test.images.reshape((-1, _time_steps, _input_units)), mnist.test.labels
print("Testing Accuracy: ", accuracy.eval(feed_dict={x: test_x, y_: test_y}))

#-----------------------------------------End of Code!---------------------------------------#
