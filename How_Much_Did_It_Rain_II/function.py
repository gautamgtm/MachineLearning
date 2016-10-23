#------------------------------------------Rainfall Prediction using RNN------------------------------------------#

#----------------------------------------- A Collection of various Functions ----------------------------------------#

#######################################################################################################################
#######################################################################################################################

#-----------------------------------------Import Libraries---------------------------------------#
import os
import glob
import time
import math
import sys
import numpy as np
from numpy import newaxis
import pandas as pd
import operator
import ConfigParser
import theano
import theano.tensor as T
import model
import keras
from keras.callbacks import ModelCheckpoint, History
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
from sklearn.cross_validation import cross_val_score, KFold, train_test_split
import matplotlib.pyplot as plt
import scipy.misc
from scipy.special import logit
import h5py

#----------------------------------------Helping Functions----------------------------------------#
def sigmoid(x):
	return 1 / (1 + math.exp(-x))

### Function to Save Model ###
def saveModel(model, name):
	json_string = model.to_json()
	open('./model/'+ name +'_model.json', 'w').write(json_string)
	model.save_weights('./model/'+ name +'_model_weights.h5', overwrite = True)
	print("Saving the Model.")

### Function to Load Model ###
def loadModel(name):
	model = model_from_json(open('./model/'+ name +'_model.json').read())
	model.load_weights('./model/best_validation_'+ name +'_model_weights.h5')
	return model
