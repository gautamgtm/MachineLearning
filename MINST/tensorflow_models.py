#-----------------------------------------Import Libraries---------------------------------------#
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import graph_util

#-----------------------------------------Tensorflow Neural Network---------------------------------------#

def FullyConnectedNN(x,_INPUT_UNITS,_HIDDEN_UNITS,_OUTPUT_UNITS):
    '''
    Arguments:
    x: an input tensor with the dimensions (N_examples, _INPUT_UNITS)
    Returns:
    A tuple (y, keep_prob).
    y is a tensor of shape (N_examples, _OUTPUT_UNITS), with values
    equal to the logits of classifying into one of _OUTPUT_UNITS classes.
    keep_prob is a scalar placeholder for the probability of
    dropout.'''
    # Fully Connected Dense Layers
    h_fc1 = tf.layers.dense(inputs=x, units=_HIDDEN_UNITS, activation=tf.nn.relu,
                    kernel_initializer=init_ops.UniformUnitScaling(), name="fc1")
    # Dropout Layer
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    out = tf.layers.dense(inputs=h_fc1_drop, units=_OUTPUT_UNITS, activation=tf.nn.relu,
                    kernel_initializer=init_ops.UniformUnitScaling(), name="out")
    return out, keep_prob


def ConvolutionalNN(x,_INPUT_UNITS,_HIDDEN_UNITS,_OUTPUT_UNITS):
    '''
    Arguments:
    x: an input tensor with the dimensions (N_examples, _INPUT_UNITS)
    Returns:
    A tuple (y, keep_prob).
    y is a tensor of shape (N_examples, _OUTPUT_UNITS), with values
    equal to the logits of classifying into one of _OUTPUT_UNITS classes.
    keep_prob is a scalar placeholder for the probability of
    dropout.'''
    # First convolutional layer - maps one grayscale image to 32 feature maps.
    h_conv1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[5, 5], strides=(1, 1),
                    padding="same", activation=tf.nn.relu, kernel_initializer=init_ops.TruncatedNormal(stddev=0.1),
                    bias_initializer=init_ops.RandomUniform(minval=0, maxval=1), name="h_conv1")

    # Pooling layer - downsamples by 2X.
    h_pool1 = tf.layers.max_pooling2d(inputs=h_conv1, pool_size=[2, 2], strides=2, padding='SAME')

    # Second convolutional layer -- maps 32 feature maps to 64.
    h_conv2 = tf.layers.conv2d(inputs=h_pool1, filters=64, kernel_size=[5, 5], strides=(1, 1),
                    padding="same", activation=tf.nn.relu, kernel_initializer=init_ops.TruncatedNormal(stddev=0.1),
                    bias_initializer=init_ops.RandomUniform(minval=0, maxval=1), name="h_conv2")

    # Second pooling layer.
    h_pool2 = tf.layers.max_pooling2d(inputs=h_conv2, pool_size=[2, 2], strides=2, padding='SAME')
    # After 2 round of downsampling, our 28x28 image is down to 7x7x64 feature maps -- maps this to 1024 features.
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

    # Fully Connected Dense Layers
    h_fc1 = tf.layers.dense(inputs=h_pool2_flat, units=_HIDDEN_UNITS, activation=tf.nn.relu,
                    kernel_initializer=init_ops.UniformUnitScaling(), name="fc1")
    # Dropout Layer
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    out = tf.layers.dense(inputs=h_fc1_drop, units=_OUTPUT_UNITS, activation=tf.nn.relu,
                    kernel_initializer=init_ops.UniformUnitScaling(), name="out")
    return out, keep_prob

def RecurrentNN(x,_input_units,_hidden_units,_output_units,_time_steps):
    '''
    Arguments:
    x: an input tensor with the dimensions (N_examples, _INPUT_UNITS)
    Returns:
    A tuple (y, keep_prob).
    y is a tensor of shape (N_examples, _OUTPUT_UNITS), with values
    equal to the logits of classifying into one of _OUTPUT_UNITS classes.
    keep_prob is a scalar placeholder for the probability of
    dropout.'''
    # A Simple RNN Layer
    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(_hidden_units, activation=tf.nn.relu)
    w_r1 = tf.Variable(tf.random_normal([_hidden_units, _output_units]), name='w_r1')
    b_r1 = tf.Variable(tf.random_normal([_output_units]), name='b_r1')

    o_r1, s_r1 = tf.nn.static_rnn(rnn_cell, tf.unstack(x, _time_steps, 1), #Unstack to get a list of 'time steps' tensors of shape (batch sz, input unit)
                    dtype=tf.float32)
    h_r1 = tf.matmul(o_r1[-1], w_r1) + b_r1
    return h_r1

#----------------------------------------- Utility Functions---------------------------------------#
def load_graph(frozen_graph_name):
    # Load the protobuf file. Parse it to retrieve the unserialized graph_def
    with tf.gfile.GFile(frozen_graph_name, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph

def freeze_graph(model_name):
    model_location = './models/'
    output_graph = model_location + model_name + "_model.pb"

    print("Creating the Tensorflow Session.")
    sess = tf.InteractiveSession()
    saver = tf.train.import_meta_graph(model_location + model_name + ".meta")
    saver.restore(sess, tf.train.latest_checkpoint("./models"))

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    output_node_names = ["accuracy"]
    output_graph_def = graph_util.convert_variables_to_constants(
                sess, # The session is used to retrieve the weights
                input_graph_def, # The graph_def is used to retrieve the nodes
                output_node_names # The output node names are used to select the usefull nodes
                )

    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())

#-----------------------------------------End of Code!---------------------------------------#
