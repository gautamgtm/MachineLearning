Projectile Trajectory Prediction using Recurrent Neural Networks

Dependencies:
* Python
* Numpy
* Pandas
* Theano
* Keras
* Matplotlib

Description:

Problem Description:
The input file contains trajectories of 100 projectiles launched at different angles and velocities. The projectiles are of unit mass and are launched from origin (0,0). Their displacement is recorded every 100ms interval. Effect of air friction is ignored and gravity is 9.8 m/s2. The data is in the following format:
[time_index] , [x] , [y]
0 , 0.0 , 0.0        # projectile 1
.
0 , 0.0 , 0.0        # projectile 2
.
0 , 0.0 , 0.0        # projectile 100
The objective is to use this data to learn how a projectile behaves

Modelling:
* Recurrent neural networks can be used to process a sequence of input vectors.
* Use Bidirectional Vanilla RNNs with sequence length 3(atleast).
* The problem can be simplified into a curve fitting problem, owing to the trajectory being of projectile. One point or even Two points(Makes a line only) would not be enough to extrapolate the next step. However, 3 or more points will be sufficient. The idea is similar to the determination of a unique circle with 3 points(Two line segments).
* The error function is chosen to be Mean Squared Error.
* To avoid overfitting, we stop the training once the loss on validation set stops decreasing. Save the model with the best validation loss.

Step 1: Processing of Raw Data
 *  Run data_preprocessing.py to generate numpy matrices of Input and Outputs of Training, Validation and Testing data.
 *  The dataset is divided into 80:20 ratio for Training to Testing samples
 *  15% of the Training Data is kept aside for Validation.
 *  The Input Vector is a sequence of length 3 depicting the last three coordinates of the projectile. The Output Vector is the next coordinates.

Step 2: Training the Recurrent Neural Network
* Define the model in the train_nn.py
* Run train_nn.py
* The models will be saved in the ./model folder

Step 3: Testing the Network and Plotting the trajectories
* Run predict_nn.py
* The test results will be in the ./results folder. Two files will be there, one for each training dataset and testing dataset.
* The files will have Expected Outputs and Predicted Outputs.
* Run plots.py
* The trajectory plots will be generated in the ./plots folder for each of the trajectories given the number of samples in the trajectory is more than 2.
* Each image will have 3 plots:
    - Actual Trajectory 
    - Predicted_1 Trajectory : It assumes that the inputs given to the network are actual coordinates. The original coordinates are used to predict the next coordinates.
    - Predicted_2 Trajectory : It potrays the actual behavior of the model. The inputs given to the network are the coordinates which are previously predicted by the model itself. The accuracy will be lower here, since the error will get accumulated with each subsequent prediction.
 

