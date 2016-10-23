#------------------------------------------Rainfall Prediction using RNN------------------------------------------#

#----------------------------------------- A Collection of various NN Models ----------------------------------------#

#######################################################################################################################
#######################################################################################################################

#-----------------------------------------Import Libraries---------------------------------------#
import operator
import theano
import theano.tensor as T
import keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Merge, Activation, Dropout, TimeDistributedDense, Flatten
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.layers.convolutional import AveragePooling1D
from keras.layers.advanced_activations import LeakyReLU


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
        left.add(LSTM(input_shape = (samples,ipt), output_dim = hid, return_sequences = False, go_backwards = False, init = 'lecun_uniform', activation='tanh'))
        right.add(LSTM(input_shape = (samples,ipt), output_dim = hid, return_sequences = False, go_backwards = True, init = 'lecun_uniform', activation='tanh'))

    if _type == 'RNN':
        left.add(SimpleRNN(input_shape = (samples,ipt), output_dim = hid, return_sequences = False, go_backwards = False, init = 'lecun_uniform', activation='tanh'))
        right.add(SimpleRNN(input_shape = (samples,ipt), output_dim = hid, return_sequences = False, go_backwards = True, init = 'lecun_uniform', activation='tanh'))

    if _type == 'GRU':
        left.add(GRU(input_shape = (samples,ipt), output_dim = hid, return_sequences = False, go_backwards = False, init = 'lecun_uniform', activation='tanh'))
        right.add(GRU(input_shape = (samples,ipt), output_dim = hid, return_sequences = False, go_backwards = True, init = 'lecun_uniform', activation='tanh'))

    model = Sequential()
    model.add(Merge([left, right], mode='concat'))
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


#----------------------------Deep Bidirectional Recurrent Neural Network-----------------------------#
'''
A Deep Bidirectional Recurrent Neural Network.
In progess.
Input =>
Output => Model
'''
def DeepBidirectionalRecurrent(_TYPE, _INPUT_UNITS, _OUTPUT_UNITS, _HIDDEN_UNITS, _LOOKBACK_SAMPLES):
    if _TYPE == 'RNN':
        ############-------------- First Layer : Bidirectional--------------############
        left, right = Sequential(), Sequential()
        left.add(SimpleRNN(input_shape = (_LOOKBACK_SAMPLES,_INPUT_UNITS), output_dim = 2*_HIDDEN_UNITS, return_sequences = True, go_backwards = False, init = 'lecun_uniform', activation='tanh'))
        right.add(SimpleRNN(input_shape = (_LOOKBACK_SAMPLES,_INPUT_UNITS), output_dim = 2*_HIDDEN_UNITS, return_sequences = True, go_backwards = True, init = 'lecun_uniform', activation='tanh'))

        model = Sequential()
        model.add(Merge([left, right], mode='concat'))
        ############-------------- Second Layer : Dense--------------############
        model.add(TimeDistributedDense(_HIDDEN_UNITS, activation = 'relu'))

        #Make two copies of Merged layer to give to Bidirectional RNN
        fork_left_1, fork_right_1 = Sequential(), Sequential()
        fork_left_1.add(model)
        fork_right_1.add(model)

        ############-------------- Third Layer : Bidirectional--------------############
        fork_left_1.add(SimpleRNN(2*_HIDDEN_UNITS, return_sequences = False, go_backwards = False, init = 'lecun_uniform', activation='tanh'))
        fork_right_1.add(SimpleRNN(2*_HIDDEN_UNITS, return_sequences = False, go_backwards = True, init = 'lecun_uniform', activation='tanh'))

        model_1 = Sequential()
        model_1.add(Merge([fork_left_1, fork_right_1], mode='concat'))
        model_1.add(Dense(output_dim = _OUTPUT_UNITS, activation = 'relu'))

    if _TYPE == 'GRU':
        ############-------------- First Layer : Bidirectional--------------############
        left, right = Sequential(), Sequential()
        left.add(GRU(input_shape = (_LOOKBACK_SAMPLES,_INPUT_UNITS), output_dim = 2*_HIDDEN_UNITS, return_sequences = True, go_backwards = False, init = 'lecun_uniform', activation='tanh'))
        right.add(GRU(input_shape = (_LOOKBACK_SAMPLES,_INPUT_UNITS), output_dim = 2*_HIDDEN_UNITS, return_sequences = True, go_backwards = True, init = 'lecun_uniform', activation='tanh'))

        model = Sequential()
        model.add(Merge([left, right], mode='concat'))
        ############-------------- Second Layer : Dense--------------############
        model.add(TimeDistributedDense(_HIDDEN_UNITS, activation = 'relu'))

        #Make two copies of Merged layer to give to Bidirectional RNN
        fork_left_1, fork_right_1 = Sequential(), Sequential()
        fork_left_1.add(model)
        fork_right_1.add(model)

        ############-------------- Third Layer : Bidirectional--------------############
        fork_left_1.add(GRU(2*_HIDDEN_UNITS, return_sequences = False, go_backwards = False, init = 'lecun_uniform', activation='tanh'))
        fork_right_1.add(GRU(2*_HIDDEN_UNITS, return_sequences = False, go_backwards = True, init = 'lecun_uniform', activation='tanh'))

        model_1 = Sequential()
        model_1.add(Merge([fork_left_1, fork_right_1], mode='concat'))
        model_1.add(Dense(output_dim = _OUTPUT_UNITS, activation = 'relu'))

    return model_1

#------------------------------------------ End of Code! ------------------------------------------#
