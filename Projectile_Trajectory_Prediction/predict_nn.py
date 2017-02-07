#------------------------------------------Projection Motion Estimation using RNN------------------------------------------#

#---------------------------------------------------- Testing of NN -------------------------------------------------#

#######################################################################################################################
#######################################################################################################################

#-----------------------------------------Import Libraries---------------------------------------#
import numpy as np
import pandas as pd
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

def predictNN(X):
    #------------------------------------------ Load RNN Model --------------------------------------------#
    print("Loading the Neural Network parameters.")
    BiRNNmodel = loadModel("BiRNN")
    predictedBiRNN = BiRNNmodel.predict([X,X], batch_size=10)
    return predictedBiRNN

#------------------------------------------Testing Model on Training Data------------------------------------------------#

print("Testing the Neural Network on Training Data.")
X_Train, Y_Train = np.load("./data/train_input.npy"), np.load("./data/train_output.npy")
predictedTrain = predictNN(X_Train)

#---------------------------------------- Creating the Output File---------------------------------------#
print("Savings the Training Results.")

#Create a new testing result file : Output file + Append the calculated outputs
Results_Train = pd.DataFrame(np.concatenate((Y_Train ,predictedTrain), axis=1))
Results_Train.to_csv("./results/training.csv", sep=',')


#------------------------------------------Testing Model on Testing Data------------------------------------------------#
print("Testing the Neural Network on Testing Data.")
X_Test, Y_Test = np.load("./data/test_input.npy"), np.load("./data/test_output.npy")
predictedTest = predictNN(X_Test)

#---------------------------------------- Creating the Output File---------------------------------------#
print("Savings the Test Results.")

### Create a new testing result file : Output file + Append the calculated outputs ###
Results_Test = pd.DataFrame(np.concatenate((Y_Test ,predictedTest), axis=1))
Results_Test.to_csv("./results/testing.csv", sep=',')

print("End of Battle! Keep Calm.")

#---------------------------------------------- End of Code! ---------------------------------------------#
