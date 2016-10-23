#------------------------------------------Rainfall Prediction using RNN------------------------------------------#

#------------------------------------------------ Preprocessing Raw Data --------------------------------------------#
'''
Data preprocessing:
1. Replaces NaN with zeros,
2. Excludes outliers,
3. Create validation holdout set
'''
#######################################################################################################################
#######################################################################################################################

#------------------------------------------Import Libraries------------------------------------------#

import glob
import os
import numpy as np
import pandas as pd
import math
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score, KFold, train_test_split

#------------------------------------------HouseKeeping-----------------------------------------#

_THRESHOLD = 73
_NFOLDS = 5
_LOOKBACK_SAMPLES = 20
_INPUT_UNITS = 22
COLUMNS = ['Id','minutes_past', 'radardist_km', 'Ref', 'Ref_5x5_10th',
       'Ref_5x5_50th', 'Ref_5x5_90th', 'RefComposite',
       'RefComposite_5x5_10th', 'RefComposite_5x5_50th',
       'RefComposite_5x5_90th', 'RhoHV', 'RhoHV_5x5_10th',
       'RhoHV_5x5_50th', 'RhoHV_5x5_90th', 'Zdr', 'Zdr_5x5_10th',
       'Zdr_5x5_50th', 'Zdr_5x5_90th', 'Kdp', 'Kdp_5x5_10th',
       'Kdp_5x5_50th', 'Kdp_5x5_90th', 'Expected']

emptyRow = np.zeros((_LOOKBACK_SAMPLES, _INPUT_UNITS))

### Extend the size of input vector to _LOOKBACK_SAMPLES * _INPUT_UNITS. Fill the empty rows with zeros ###
def extendSeries(arr):
    curr_len = arr.shape[0]
    extra_needed = _LOOKBACK_SAMPLES-curr_len
    if (extra_needed > 0):
        arr = np.concatenate((arr, emptyRow[0:extra_needed,:]), axis=0)
    return arr

def createTrainDataset(raw_data):
    grouped = raw_data.groupby('Id')
    data_size = len(grouped)
    tempX, tempY = np.empty((data_size, _LOOKBACK_SAMPLES,_INPUT_UNITS)), np.empty((data_size,1))
    nIter = 0
    for _, group in grouped:
        group_array = np.array(group)
        X = extendSeries(group_array[:,1:23])
        Y = np.array(group_array[0,23])
        tempX[nIter,:,:], tempY[nIter,:] = X[:, :], Y
        nIter += 1
    return tempX, tempY

def createTestDataset(raw_data):
    grouped = raw_data.groupby('Id')
    data_size = len(grouped)
    tempX = np.empty((data_size, _LOOKBACK_SAMPLES,_INPUT_UNITS))
    nIter = 0
    for _, group in grouped:
        group_array = np.array(group)
        X = extendSeries(group_array[:,1:23])
        tempX[nIter,:,:] = X[:, :]
        nIter += 1
    return tempX

#------------------------------------------Training and Validation Data------------------------------------------#

#Import the training data
print('Reading the list of training data files...')
train_raw = pd.read_csv("./data/train.csv")
raw_ids_all = train_raw["Id"]
raw_ids = raw_ids_all.unique()

####### 2. Remove ids with only NaNs in the "Ref" column #######
train_raw_tmp = train_raw[~np.isnan(train_raw.Ref)]
raw_ids_tmp = train_raw_tmp["Id"].unique()
train_new = train_raw[np.in1d(raw_ids_all, raw_ids_tmp)]

####### 3. Convert all NaN to zero #######
train_new = train_new.fillna(0.0)
train_new = train_new.reset_index(drop=True)

X_Train, Y_Train = createTrainDataset(train_new)

####### 4. Define and exclude outliers from training set #######
meaningful_ids = (Y_Train < _THRESHOLD).nonzero()[0]
X_Train, Y_Train = X_Train[meaningful_ids], Y_Train[meaningful_ids]

print("Splitting the Data into Training and Validation Set.")
kf = KFold(len(X_Train), n_folds = _NFOLDS)
idx = 0
for train, test in kf:
    X_Validation_, Y_Validation_ = X_Train[test], Y_Train[test]
    np.save("./data/processed_train/validationInput%s" % (i), np.array(X_Validation_))
    np.save("./data/processed_train/validationOutput%s" % (i), np.array(Y_Validation_))
    print("Converted the Validation Data Input to pass to the NN for prediction at every sample. Shape of Validation data Input: ",X_Validation_.shape)
    print("Converted the Validation Data Output to pass to the NN for prediction at every sample. Shape of Validation data Output: ",Y_Validation_.shape)
    X_Train_, Y_Train_ = X_Train[train], Y_Train[train]
    np.save("./data/processed_train/trainInput%s" % (i), np.array(X_Train_))
    np.save("./data/processed_train/trainOutput%s" % (i), np.array(Y_Train_))
    print("Converted the Training Data Input to pass to the NN for prediction at every sample. Shape of Training data Input: ",X_Train_.shape)
    print("Converted the Training Data Output to pass to the NN for prediction at every sample. Shape of Training data Output: ",Y_Train_.shape)
    idx = idx + 1

#------------------------------------------Testing Data------------------------------------------#

#Import the testing data
print('Reading the list of testing data files...')
test_raw = pd.read_csv("./data/test.csv")
test_raw_ids_all = test_raw["Id"]
test_raw_ids = np.array(test_raw_ids_all.unique())

# Convert all NaNs to zero
test_new = test_raw.fillna(0.0)
test_new = test_new.reset_index(drop=True)

X_Test = createTestDataset(test_new)

np.save("./data/processed_test/testInput", np.array(X_Test))

print("Converted the Testing Data Input to pass to the NN for prediction at every sample. Shape of Testing data Input: ",X_Test.shape)

#------------------------------------------ End of Code! ------------------------------------------#
