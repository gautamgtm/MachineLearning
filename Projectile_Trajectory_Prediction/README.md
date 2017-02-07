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
    - Predicted_1 Trajectory : It assumes that the inputs given to the network are actual coordinates.
    - Predicted_2 Trajectory : It potrays the actual behavior of the model. The inputs given to the network are the coordinates which are previously predicted by the model intself.

