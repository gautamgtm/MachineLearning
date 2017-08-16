#-----------------------------------------Import Libraries---------------------------------------#
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import graph_util
from tensorflow_models import ConvolutionalNN, FullyConnectedNN, freeze_graph, load_graph

#-----------------------------------------Test NN---------------------------------------#

#-----------------------------------------Import Data---------------------------------------#
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#-----------------------------------------Import Graph from Protobuf File---------------------------------------#
graph = load_graph("./models/mnist_model.pb")

# List of Operations in the graph
#for op in graph.get_operations():
#    print(op.name)

x = graph.get_tensor_by_name('prefix/input:0')
y_ = graph.get_tensor_by_name('prefix/output:0')
keep_prob = graph.get_tensor_by_name('prefix/keep_prob:0')
accuracy = graph.get_tensor_by_name('prefix/accuracy:0')

#-----------------------------------------Testing Model---------------------------------------#
print("Creating the Tensorflow Session.")
sess = tf.InteractiveSession(graph=graph)

print("Testing the Neural Network.")
print("Testing Accuracy: ", accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

#-----------------------------------------End of Code!---------------------------------------#
