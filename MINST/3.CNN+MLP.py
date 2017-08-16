#-----------------------------------------Import Libraries---------------------------------------#
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import init_ops
from tensorflow_models import ConvolutionalNN, FullyConnectedNN, freeze_graph, load_graph

#-----------------------------------------Model Parameters----------------------------------------#
_INPUT_UNITS = 784
_HIDDEN_UNITS = 256
_OUTPUT_UNITS = 10
_LEARNING_RATE = 1
_NEPOCH = 5000
_BATCH_SIZE = 50

#-----------------------------------------Import Data---------------------------------------#
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#-----------------------------------------Create Model---------------------------------------#
print("Creating a Neural Network.")
random.seed(123)

# Define Inputs and Outputs
x = tf.placeholder(tf.float32, [None, _INPUT_UNITS], name="input")
y_ = tf.placeholder(tf.float32, [None, _OUTPUT_UNITS], name="output")

# Define Neural Network Layers
# Reshape the input to use within a convolutional neural net.
x_image = tf.reshape(x, [-1,28,28,1], name="x_image")
# Convolutional Layers
#h_fc2, keep_prob = ConvolutionalNN(x_image,_INPUT_UNITS,_HIDDEN_UNITS,_OUTPUT_UNITS)
# First convolutional layer - maps one grayscale image to 32 feature maps.
h_conv1 = tf.layers.conv2d(inputs=x_image, filters=32, kernel_size=[5, 5], strides=(1, 1),
                padding="same", activation=tf.nn.relu, kernel_initializer=init_ops.TruncatedNormal(stddev=0.1),
                bias_initializer=init_ops.RandomUniform(minval=0, maxval=1), name="h_conv1")

# Pooling layer - downsamples by 2X.
h_pool1 = tf.layers.max_pooling2d(inputs=h_conv1, pool_size=[2, 2], strides=2, padding='SAME', name="h_pool1")

# Second convolutional layer -- maps 32 feature maps to 64.
h_conv2 = tf.layers.conv2d(inputs=h_pool1, filters=64, kernel_size=[5, 5], strides=(1, 1),
                padding="same", activation=tf.nn.relu, kernel_initializer=init_ops.TruncatedNormal(stddev=0.1),
                bias_initializer=init_ops.RandomUniform(minval=0, maxval=1), name="h_conv2")

# Second pooling layer.
h_pool2 = tf.layers.max_pooling2d(inputs=h_conv2, pool_size=[2, 2], strides=2, padding='SAME', name="h_pool2")
# After 2 round of downsampling, our 28x28 image is down to 7x7x64 feature maps -- maps this to 1024 features.
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64], name="h_pool2_flat")

# Fully Connected Dense Layers
h_fc1 = tf.layers.dense(inputs=h_pool2_flat, units=_HIDDEN_UNITS, activation=tf.nn.relu,
                kernel_initializer=init_ops.UniformUnitScaling(), name="h_fc1")
# Dropout Layer
keep_prob = tf.placeholder(tf.float32, name="keep_prob")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name="h_fc1_drop")

h_out = tf.layers.dense(inputs=h_fc1_drop, units=_OUTPUT_UNITS, activation=tf.nn.relu,
                kernel_initializer=init_ops.UniformUnitScaling(), name="h_out")

#-----------------------------------------Define Loss and Optimizer---------------------------------------#
print("Compiling the Neural Network.")
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=h_out), name="loss")
train_step = tf.train.AdadeltaOptimizer(_LEARNING_RATE).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(h_out,1), tf.argmax(y_,1)), tf.float32), name="accuracy")

#-----------------------------------------Create TF Session---------------------------------------#
print("Creating the Tensorflow Session.")
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
saver = tf.train.Saver(max_to_keep=25)

#-----------------------------------------Training Model---------------------------------------#
print("Training the Neural Network : Started.")
max_val_accuracy, patience, cnt = 0, 500, 0
for i in range(_NEPOCH):
    batch_xs, batch_ys = mnist.train.next_batch(_BATCH_SIZE)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
    # Monitor Validation Loss (Every 10 epchs) to save the best model and early stopping
    cnt = cnt + 1
    if cnt == patience:
        break
    if i%50 == 0:
        val_accuracy = accuracy.eval(feed_dict={x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0})
        train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
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
print("Testing Accuracy: ", accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

#-----------------------------------------End of Code!---------------------------------------#
