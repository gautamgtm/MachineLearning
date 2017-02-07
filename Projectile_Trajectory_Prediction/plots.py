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
    predictedBiRNN = BiRNNmodel.predict([X,X], batch_size=10)
    return predictedBiRNN

def handleTestCase(X, y):
    ### yth Projectile Motion ###
    ### PLotting Loss against epoch ###
    plt.title('Projectile %s' % (y+1))
    plt.ylabel('Y Axis')
    plt.xlabel('X Axis')
    ### Only 1 data point found, return ###
    if(X.shape[0]<=2):
        return
    ### Plot Original Trajectory ###
    x1, y1 = X[:,1], X[:,2]
    plt.plot(x1, y1, linewidth=0.5, color='blue', linestyle='--', marker='o', label='Original')
    ### Predicted_1 : Use original locations to predict next location ###
    x2, y2 = [x1[0],x1[1]], [y1[0],y1[1]]
    ### Predict Trajecory ###
    ### 1st Predction ###
    a, b = X[0][1:], X[1][1:]
    input_data = np.vstack((a,a,b)).reshape(1,3,2)
    predicted_output = predictNN(input_data)
    x2.append(predicted_output[0][0])
    y2.append(predicted_output[0][1])
    ### Predicted_2 : Use previous predictions to predict next location ###
    x3, y3 = [x1[0],x1[1],x2[2]], [y1[0],y1[1], y2[2]]
    ### Rest of Predictions : Predicted_1 ###
    for i in range(2,X.shape[0]-1):
        input_data = np.vstack((X[i-2][1:],X[i-1][1:],X[i][1:])).reshape(1,3,2)
        predicted_output = predictNN(input_data)
        x2.append(predicted_output[0][0])
        y2.append(predicted_output[0][1])
    ### Rest of Predictions : Predicted_2 ###
    for i in range(2,X.shape[0]-1):
        input_data = np.vstack(([x3[i-2],y3[i-2]],[x3[i-1],y3[i-1]],[x3[i],y3[i]])).reshape(1,3,2)
        predicted_output = predictNN(input_data)
        x3.append(predicted_output[0][0])
        y3.append(predicted_output[0][1])
    ### Plot Predicted Trajectory ###
    plt.plot(x2, y2, linewidth=1.0, color='green', linestyle='--', marker='*', label='Predicted_1')
    plt.plot(x3, y3, linewidth=2.0, color='red', linestyle='--', marker='v', label='Predicted_2')
    plt.legend()
    plt.savefig("./plots/Projectile_Plot_%s.png" % (y+1))
    plt.clf()

#------------------------------------------ Load RNN Model --------------------------------------------#
print("Loading the Neural Network parameters.")
BiRNNmodel = loadModel("BiRNN")

#------------------------------------------Testing Model on Data------------------------------------------------#

#Import the raw data
print('Reading the data file...')
df = np.loadtxt("./data/projectiles.csv", delimiter=',')
projCount, idx = 0, 0
while idx<df.shape[0]:
    motion = df[idx].reshape(1,3)
    idx = idx+1
    while df[idx][0] != 0:
        motion = np.append(motion, df[idx].reshape(1,3), axis=0)
        idx = idx+1
        if idx==df.shape[0]:
            break
    handleTestCase(motion, projCount)
    projCount = projCount + 1

print projCount

#------------------------------------------Testing Model on New Use Case------------------------------------------------#
#Predict the trajectory of a projectile launched at 45 degrees with an initial velocity of 10 m/s till it hits the ground or time_index=100 whichever is earlier.#
#You can assume the initial two points in the trajectory to be :
#0 ,0.0 ,0.0
#1 ,0.707106781187 ,0.658106781187#

### PLotting Loss against epoch ###
plt.title('Projectile 000')
plt.ylabel('Y Axis')
plt.xlabel('X Axis')
### 1st Predction ###
x, y = [0,0.707106781187], [0,0.658106781187]
input_data = np.vstack(([0,0],[0,0],[0.707106781187 ,0.658106781187])).reshape(1,3,2)
predicted_output = predictNN(input_data)
x.append(predicted_output[0][0])
y.append(predicted_output[0][1])
### Rest of Predictions : Predicted_2 ###
for i in range(2,105):
    if(y[len(y)-1] < 0.5):
        break
    input_data = np.vstack(([x[i-2],y[i-2]],[x[i-1],y[i-1]],[x[i],y[i]])).reshape(1,3,2)
    predicted_output = predictNN(input_data)
    x.append(predicted_output[0][0])
    y.append(predicted_output[0][1])
### Plot Predicted Trajectory ###
location = [[0,0]]
for i in range(len(x)):
    location.append([x[i], y[i]])
print location
np.savetxt("./results/example.csv", np.asarray(location), delimiter=",")
plt.plot(x, y, linewidth=2.0, color='red', linestyle='--', marker='v', label='Predicted_2')
plt.legend()
plt.savefig("./plots/Projectile_Plot_000.png")
plt.clf()


print("End of Battle! Keep Calm.")

#---------------------------------------------- End of Code! ---------------------------------------------#
