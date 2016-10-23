#------------------------------------------Rainfall Prediction using RNN------------------------------------------#

#---------------------------------------------------- Training of NN ------------------------------------------------#

#######################################################################################################################
#######################################################################################################################

#-----------------------------------------Import Libraries---------------------------------------#
import os
import glob
import time
import sys
import numpy as np
import pandas as pd
import operator
import ConfigParser
import theano
import theano.tensor as T
import function
import keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Merge, Activation, Dropout, TimeDistributedDense
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.callbacks import ModelCheckpoint, History
from keras.optimizers import SGD
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from model import Simple, Bidirectional, DeepRecurrent, DeepBidirectionalRecurrent
from function import saveModel, loadModel
import matplotlib.pyplot as plt

print("Brace Yourself! Winter is Coming!")

#######################################################################################################################
def trainNN(i):
    #-----------------------------------------Prepare Data----------------------------------------#
    X_Train, Y_Train, X_Validation, Y_Validation = np.load("./data/processed_train/trainInput%s.npy" % (i)), np.load("./data/processed_train/trainOutput%s.npy" % (i)), np.load("./data/processed_train/validationInput%s.npy" % (i)), np.load("./data/processed_train/validationOutput%s.npy" % (i))

    #------------------------------------------ RNN Model ---------------------------------------------#

    ### Create a Neural Network ###
    print("Creating a Neural Network.")

    BiRNNmodel = Bidirectional('RNN', _INPUT_UNITS, _OUTPUT_UNITS, _HIDDEN_UNITS, _LOOKBACK_SAMPLES)
    BiGRUmodel = Bidirectional('GRU', _INPUT_UNITS, _OUTPUT_UNITS, _HIDDEN_UNITS, _LOOKBACK_SAMPLES)
    #BiRNNmodel = DeepRecurrent(['RNN', 'Simple', 'RNN'], [2*_HIDDEN_UNITS, _HIDDEN_UNITS, 2*_HIDDEN_UNITS], _INPUT_UNITS, _OUTPUT_UNITS, _LOOKBACK_SAMPLES)
    #BiGRUmodel = DeepRecurrent(['GRU', 'Simple', 'GRU'], [2*_HIDDEN_UNITS, _HIDDEN_UNITS, 2*_HIDDEN_UNITS], _INPUT_UNITS, _OUTPUT_UNITS, _LOOKBACK_SAMPLES)


    #------------------------------------------Training Model------------------------------------------#

    ### Stochastic Gradient Descent Optimizer ###
    sgd = SGD(lr = 0.02, decay = 1e-6, momentum = 0.9, nesterov = True)

    ### Compile the NN with Loss calculation and Optimizer choice ###
    print("Compiling the Neural Network.")
    BiGRUmodel.compile(loss="mse", optimizer="adagrad")
    BiRNNmodel.compile(loss="mse", optimizer="adadelta")

    #--------------------------------------------Callbacks---------------------------------------------#

    ### Saves the model weights after each epoch if the validation loss decreased ###
    checkpointerBiGRU = ModelCheckpoint(filepath="./model/best_validation_BiGRU%s_model.h5" % (i), verbose=1, save_best_only=True)
    checkpointerBiRNN = ModelCheckpoint(filepath="./model/best_validation_BiRNN%s_model.h5" % (i), verbose=1, save_best_only=True)

    ### Stop training when a monitored quantity has stopped improving ###
    stopEarly = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='auto')

    ### Save Model History ###
    hist = History()

    ### Train the NN using Cross Validation methods ###
    print("Training the Neural Network : Started.")
    BiRNNmodel.fit(X_Train, Y_Train, batch_size=1000, nb_epoch=_NEPOCH, shuffle = True, validation_data=(X_Validation, Y_Validation), callbacks=[checkpointerBiRNN, hist, stopEarly])
    BiGRUmodel.fit(X_Train, Y_Train, batch_size=1000, nb_epoch=_NEPOCH, shuffle = True, validation_data=(X_Validation, Y_Validation), callbacks=[checkpointerBiGRU, hist, stopEarly])

    ### Training Completed ###
    print("Training the Neural Network : Completed.")


#######################################################################################################################

#-----------------------------------------Model Parameters----------------------------------------#
print("Loading the Model Parameters.")
### Load the Model parameters from Conf file. ###
config = ConfigParser.ConfigParser()
config.readfp(open(r'conf.ini'))

_INPUT_UNITS = int(config.get('Model', '_INPUT_UNITS'))
_HIDDEN_UNITS = int(config.get('Model', '_HIDDEN_UNITS'))
_OUTPUT_UNITS = int(config.get('Model', '_OUTPUT_UNITS'))
_LEARNING_RATE = float(config.get('Model', '_LEARNING_RATE'))
_NEPOCH = int(config.get('Model', '_NEPOCH'))
_LOOKBACK_SAMPLES = int(config.get('Model', '_LOOKBACK_SAMPLES'))

for idx in range(5):
    trainNN(idx)

print("End of Battle! Keep Calm.")

#---------------------------------------------- End of Code! ---------------------------------------------#
