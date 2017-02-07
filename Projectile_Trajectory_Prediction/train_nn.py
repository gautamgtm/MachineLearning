#------------------------------------------Projection Motion Estimation using RNN------------------------------------------#

#---------------------------------------------------- Training of NN ------------------------------------------------#

#######################################################################################################################
#######################################################################################################################

#-----------------------------------------Import Libraries---------------------------------------#
import numpy as np
import operator
import theano
import theano.tensor as T
import keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Merge, Activation, Dropout, TimeDistributedDense
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.callbacks import ModelCheckpoint, History
from keras.optimizers import SGD, Adagrad
from model import Simple, Bidirectional, DeepRecurrent, saveModel, loadModel
import matplotlib.pyplot as plt


#-----------------------------------------Model Parameters----------------------------------------#
_INPUT_UNITS = 2
_HIDDEN_UNITS = 40
_OUTPUT_UNITS = 2
_LEARNING_RATE = 0.01
_NEPOCH = 1000
_LOOKBACK_SAMPLES = 3


#######################################################################################################################
#-----------------------------------------Import Data----------------------------------------#
X_Train, Y_Train, X_Validation, Y_Validation = np.load("./data/train_input.npy"), np.load("./data/train_output.npy"), np.load("./data/val_input.npy"), np.load("./data/val_output.npy")

#------------------------------------------ RNN Model ---------------------------------------------#

### Create a Neural Network ###
print("Creating a Neural Network.")

BiRNNmodel = Bidirectional('RNN', _INPUT_UNITS, _OUTPUT_UNITS, _HIDDEN_UNITS, _LOOKBACK_SAMPLES)

#------------------------------------------Training Model------------------------------------------#

### Compile the NN with Loss calculation and Optimizer choice ###
print("Compiling the Neural Network.")
BiRNNmodel.compile(loss="mse", optimizer="adadelta")

#--------------------------------------------Callbacks---------------------------------------------#

### Saves the model weights after each epoch if the validation loss decreased ###
checkpointerBiRNN = ModelCheckpoint(filepath="./model/best_validation_BiRNN_model.h5", verbose=1, save_best_only=True)

### Stop training when a monitored quantity has stopped improving ###
stopEarly = keras.callbacks.EarlyStopping(monitor='val_loss', patience=250, verbose=0, mode='auto')

### Train the NN using Cross Validation methods ###
print("Training the Neural Network : Started.")
BiRNNmodel.fit([X_Train, X_Train], Y_Train, batch_size=5, nb_epoch=_NEPOCH, shuffle = True, validation_data=([X_Validation, X_Validation], Y_Validation), callbacks=[checkpointerBiRNN, stopEarly])

### Training Completed ###
print("Training the Neural Network : Completed.")

#######################################################################################################################

#---------------------------------------------- End of Code! ---------------------------------------------#
