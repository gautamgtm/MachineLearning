#----------------------------------------- A Collection of various NN Models ----------------------------------------#

#######################################################################################################################
#######################################################################################################################

#-----------------------------------------Import Libraries---------------------------------------#
import operator
import theano
import theano.tensor as T
import keras
from keras.models import Sequential, load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Merge, Activation, Dropout, TimeDistributedDense, Flatten
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.layers.convolutional import AveragePooling1D
from keras.layers.advanced_activations import LeakyReLU

### Function to Save Model ###
def saveModel(model, name):
	json_string = model.to_json()
	open('./model/'+ name +'_model.json', 'w').write(json_string)
	model.save_weights('./model/'+ name +'_model_weights.h5', overwrite = True)
	print("Saving the Model.")

### Function to Load Model ###
def loadModel(name):
	#model = model_from_json(open('./model/'+ name +'_model.json').read())
	model = load_model('./model/best_validation_'+ name +'_model.h5')
	return model

#--------------------------------------Simple Neural Network---------------------------------------#
'''
A Simple Neural Network.
Input => layers : An array of size = num of layers and elements = number of nodes in the layer.
         func : An array of size = num of layers and elements = activation function of the layer.
         ipt : number of inputs.
Output => Model
'''
def Simple(layers, func, ipt):
    model = Sequential()
    #model.add(BatchNormalization(input_shape = [ipt]))
    model.add(Dense(layers[0], input_dim = ipt, activation = func[0]))
    for i in range(1, len(layers)):
        model.add(Dense(layers[i], activation = func[i]))
    return model



#------------------------------Bidirectional Recurrent Neural Network-------------------------------#
'''
A Bidirectional Recurrent Neural Network.
Input => _type :  _type of RNN.
         ipt : number of inputs.
         opt : number of outputs.
         hid : number of nodes in RNN
         samples : length of input sequence, i.e. _LOOKBACK_SAMPLES
Output => Model
'''
def Bidirectional(_type, ipt, opt, hid, samples):
    left, right = Sequential(), Sequential()
    if _type == 'LSTM':
        left.add(LSTM(input_shape = (samples,ipt), output_dim = hid, return_sequences = False, go_backwards = False, init = 'lecun_uniform', activation='relu'))
        right.add(LSTM(input_shape = (samples,ipt), output_dim = hid, return_sequences = False, go_backwards = True, init = 'lecun_uniform', activation='relu'))

    if _type == 'RNN':
        left.add(SimpleRNN(input_shape = (samples,ipt), output_dim = hid, return_sequences = False, go_backwards = False, init = 'lecun_uniform', activation='relu'))
        right.add(SimpleRNN(input_shape = (samples,ipt), output_dim = hid, return_sequences = False, go_backwards = True, init = 'lecun_uniform', activation='relu'))

    if _type == 'GRU':
        left.add(GRU(input_shape = (samples,ipt), output_dim = hid, return_sequences = False, go_backwards = False, init = 'lecun_uniform', activation='relu'))
        right.add(GRU(input_shape = (samples,ipt), output_dim = hid, return_sequences = False, go_backwards = True, init = 'lecun_uniform', activation='relu'))

    model = Sequential()
    model.add(Merge([left, right], mode='concat'))
    model.add(Dense(0.5*hid, activation = 'relu'))
    model.add(Dense(output_dim = opt, activation = 'relu'))
    return model


#----------------------------------Deep Recurrent Neural Network-----------------------------------#
'''
A Deep Recurrent Neural Network.
Input => _type : An array of size = num of layers and elements = _type of layer
         layers : An array of size = num of layers and elements = number of nodes in the layer.
         func : An array of size = num of layers and elements = activation function of the layer.
         ipt : number of inputs.
         opt : number of outputs.
         samples : length of input sequence, i.e. _LOOKBACK_SAMPLES
Output => Model
'''
def DeepRecurrent(_type, layers, ipt, opt, samples):
    model = Sequential()

    ### First Layer ###
    if _type[0] == 'Simple':
        model.add(TimeDistributedDense(layers[0], input_dim = ipt, input_length = samples))
    if _type[0] == 'LSTM':
        model.add(LSTM(input_shape = (samples, ipt), output_dim = layers[0], return_sequences = True))
    if _type[0] == 'RNN':
        model.add(SimpleRNN(input_shape = (samples, ipt), output_dim = layers[0], return_sequences = True))
    if _type[0] == 'GRU':
        model.add(GRU(input_shape = (samples, ipt), output_dim = layers[0], return_sequences = True))

    ### Next Layers ###
    for i in range(1, len(layers)-1):
        if _type[0] == 'Simple':
            model.add(TimeDistributedDense(layers[i]))
        if _type[0] == 'LSTM':
            model.add(LSTM(layers[i], return_sequences = True))
        if _type[0] == 'RNN':
            model.add(SimpleRNN(layers[i], return_sequences = True))
        if _type[0] == 'GRU':
            model.add(GRU(layers[i], return_sequences = True))

    ### Last Layer ###
    if _type[0] == 'Simple':
        model.add(TimeDistributedDense(layers[len(layers)-1]))
    if _type[0] == 'LSTM':
        model.add(LSTM(layers[len(layers)-1], return_sequences = False))
    if _type[0] == 'RNN':
        model.add(SimpleRNN(layers[len(layers)-1], return_sequences = False))
    if _type[0] == 'GRU':
        model.add(GRU(layers[len(layers)-1], return_sequences = False))

    ### Output Layer ###
    model.add(Dense(output_dim = opt, activation = 'linear'))
    return model


#------------------------------------------ End of Code! ------------------------------------------#
