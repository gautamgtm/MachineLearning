Projectile Trajectory Prediction using Recurrent Neural Networks

Dependencies:
* Python
* Numpy
* Pandas
* Theano
* Keras
* Matplotlib

Description:
Step 1: Processing of Raw Data
 *  Run data_preprocessing.py to generate numpy matrices of Input and Outputs of Training, Validation and Testing data.
 *  The dataset is divided into 80:20 ratio for Training to Testing samples
 *  15% of the Training Data is kept aside for Validation.
 *  The Input Vector is a sequence of length 3 depicting the last three coordinates of the projectile. The Output Vector is the next coordinates.

Step 2: Training the Recurrent Neural Network


