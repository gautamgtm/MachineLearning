#------------------------------------------Rainfall Prediction using RNN------------------------------------------#

#---------------------------------------------------- Testing of NN -------------------------------------------------#

#######################################################################################################################
#######################################################################################################################

#-----------------------------------------Import Libraries---------------------------------------#
import os
import glob
import time
import sys
from math import exp, sqrt
import numpy as np
import pandas as pd
import operator
import ConfigParser
import theano
import theano.tensor as T
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD
from keras.models import model_from_json
from function import saveModel, loadModel
import matplotlib.pyplot as plt

print("Brace Yourself! Winter is Coming!")


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

#-----------------------------------------Prepare Data----------------------------------------#
X_Train, Y_Train, X_Validation, Y_Validation = np.load("./data/processed_train/trainInput.npy"), np.load("./data/processed_train/trainOutput.npy"), np.load("./data/processed_train/validationInput.npy"), np.load("./data/processed_train/validationOutput.npy")
X_Test = np.load("./data/processed_test/testInput.npy")

#------------------------------------------ Load RNN Model --------------------------------------------#
print("Loading the Neural Network parameters.")
BiGRUmodel = loadModel("BiGRU")
BiRNNmodel = loadModel("BiRNN")

#----------------------------------------- Compile RNN Model ------------------------------------------#

### Stochastic Gradient Descent Optimizer ###
sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)

### Compile the NN with Loss calculation and Optimizer choice ###
print("Compiling the Neural Network.")
BiGRUmodel.compile(loss="mse", optimizer="adagrad")
BiRNNmodel.compile(loss="mse", optimizer="adadelta")

#------------------------------------------Testing Model on Training Data------------------------------------------------#
print("Testing the Neural Network on Training Data.")
predictedBiGRU = BiGRUmodel.predict([X_Train, X_Train], batch_size=100)
predictedBiRNN = BiRNNmodel.predict([X_Train, X_Train], batch_size=100)
predicted = (predictedBiRNN+predictedBiGRU)/2
mae = np.absolute(((predicted - Y_Train)).mean(axis=0))
print("Mean Absolute Error on Training Data : " , mae)

#---------------------------------------- Creating the Output File---------------------------------------#
print("Savings the Training Results.")

#Create a new testing result file : Output file + Append the calculated outputs
idx = [i for i in range(1,len(X_Train)+1)]
idx = np.array(idx).reshape(-1,1)
Results_Train = pd.DataFrame(np.concatenate((idx, Y_Train ,predicted), axis=1), columns=['Id','Actual','Expected'])
Results_Train.to_csv("./results/training.csv", sep=',', float_format='%.15f')

#------------------------------------------Testing Model on Validation Data------------------------------------------------#
print("Testing the Neural Network on Validation Data.")
predictedBiGRU = BiGRUmodel.predict([X_Validation, X_Validation], batch_size=100)
predictedBiRNN = BiRNNmodel.predict([X_Validation, X_Validation], batch_size=100)
predicted = (predictedBiRNN+predictedBiGRU)/2
mae = np.absolute(((predicted - Y_Validation)).mean(axis=0))
print("Mean Absolute Error on Validation Data : " , mae)

#---------------------------------------- Creating the Output File---------------------------------------#
print("Savings the Validation Results.")

#Create a new testing result file : Output file + Append the calculated outputs
idx = [i for i in range(1,len(X_Validation)+1)]
idx = np.array(idx).reshape(-1,1)
Results_Train = pd.DataFrame(np.concatenate((idx, Y_Validation ,predicted), axis=1), columns=['Id','Actual','Expected'])
Results_Train.to_csv("./results/validation.csv", sep=',', float_format='%.15f')

#------------------------------------------Testing Model on Testing Data------------------------------------------------#
print("Testing the Neural Network on Testing Data.")
predictedBiGRU = BiGRUmodel.predict([X_Test, X_Test], batch_size=100)
predictedBiRNN = BiRNNmodel.predict([X_Test, X_Test], batch_size=100)
predicted = (predictedBiRNN+predictedBiGRU)/2

#---------------------------------------- Creating the Output File---------------------------------------#
print("Savings the Test Results.")

### Create a new testing result file : Output file + Append the calculated outputs ###
idx = [i for i in range(1,len(X_Test)+1)]
idx = np.array(idx).reshape(-1,1)
Results_Test = pd.DataFrame(np.concatenate((idx, predicted), axis=1), columns=['Id','Expected'])
Results_Test.to_csv("./results/testing.csv", sep=',', float_format='%.15f')

print("End of Battle! Keep Calm.")

#---------------------------------------------- End of Code! ---------------------------------------------#
