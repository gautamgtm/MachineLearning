#----------------------------------------- Preprocessing Raw Data  ----------------------------------------#

#######################################################################################################################
#######################################################################################################################

#------------------------------------------Import Libraries------------------------------------------#

import numpy as np
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score, KFold, train_test_split

#------------------------------------------Import Data------------------------------------------#

#Import the training data
print('Reading the data file...')
df = np.loadtxt("./data/projectiles.csv", delimiter=',')
input_data, output_data = np.empty((1,3,2)), np.empty((1,2))
idx = 0
while idx+3<df.shape[0]:
	### 1st input vector of a projectile motion ###
	if df[idx][0] == 0:
		a, b = df[idx][1:].reshape(1,2), df[idx+1][1:].reshape(1,2)
		input_data = np.append(input_data, np.vstack((a,a,b)).reshape(1,3,2), axis=0)
		output_data = np.append(output_data, df[idx+2][1:].reshape(1,2), axis=0)
	if df[idx+3][0] > 0:
		a, b, c = df[idx][1:].reshape(1,2), df[idx+1][1:].reshape(1,2), df[idx+2][1:].reshape(1,2)
		input_data = np.append(input_data, np.vstack((a,b,c)).reshape(1,3,2), axis=0)
		output_data = np.append(output_data, df[idx+3][1:].reshape(1,2), axis=0)
		idx = idx+1
	else:
		idx = idx+3
input_data, output_data = np.delete(input_data, (0), axis=0), np.delete(output_data, (0), axis=0)

for i in range(10):
	print input_data[i], output_data[i]

msk = np.random.rand(len(input_data)) < 0.8

train_input, train_output = input_data[msk], output_data[msk]
test_input, test_output = input_data[~msk], output_data[~msk]

msk = np.random.rand(len(train_input)) < 0.85
val_input, val_output = train_input[~msk], train_output[~msk]
train_input, train_output = train_input[msk], train_output[msk]

np.save("./data/train_input.npy", train_input)
np.save("./data/train_output.npy", train_output)
np.save("./data/val_input.npy", val_input)
np.save("./data/val_output.npy", val_output)
np.save("./data/test_input.npy", test_input)
np.save("./data/test_output.npy", test_output)

print train_input.shape, train_output.shape, val_input.shape, val_output.shape, test_input.shape, test_output.shape

print('Saved the processed training data...')

#------------------------------------------ End of Code! ------------------------------------------#
