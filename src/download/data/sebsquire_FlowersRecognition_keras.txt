# FlowersRecognition_keras
Multi-class Image Classification using CNN (Flower Types) (Keras, Tensorflow-GPU, OpenCV) 
Dataset available at: https://www.kaggle.com/alxmamaev/flowers-recognition

Includes:
 - Keras CNN implemented using keras.models.Sequential
 - Keras CNN incorporating a residual block/shortcut connection
 - Hyperparameter tuning using grid search
 - Learning rate annealing using keras.optimizers.adadelta
 - Dropout in fully connected layer to reduce overfit
 - Normalisation of image data
 
Modules:
 - prepocessing: Function for creation of training and test data in .npy files.
 - main_keras: Calls preprocessing.py to load and preprocess image data, sets model parameters, loads and normalises data, and trains, tests, and produces classification metrics for the convolutional neural network model on the validation set using a sequential model.
 - hyperparamsearch: Calls preprocessing.py to load and preprocess image data, sets model parameters, loads and normalises data, and trains, tests, and produces classification metrics for the convolutional neural network model on the validation set using a sequential model. Performs a grid search of hyperparameter options and records accuracy and epoch number of max accuracy in a separate csv.
 - res_hyperparamsearch: Calls preprocessing.py to load and preprocess image data, sets model parameters, loads and normalises data, and trains, tests, and produces classification metrics for the convolutional neural network model with residual block/shortcut connection on the validation set using a sequential model. Performs a grid search of hyperparameter options and records accuracy and epoch number of max accuracy in a separate csv.

Info on residual layers:
    ResNet-50:
        - Deep Residual Learning for Image Recognition - https://arxiv.org/pdf/1512.03385.pdf

Python 3.6.7, Tensorflow 1.12.0, Keras 2.2.4
