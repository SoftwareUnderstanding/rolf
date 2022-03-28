# Dogs-and-cats-image-classification-CNN
Binary Image Classification using CNN w/ residual layers (Dogs &amp; Cats)
(Tensorflow-GPU, TFLearn, OpenCV)

Modules converted from Sentdex' tutorial (https://pythonprogramming.net/convolutional-neural-network-kats-vs-dogs-machine-learning-tutorial/). Now includes:
 - Improved accuracy through CNN architecture and hyperparameter modification featuring inclusion of residual blocks
(Ref: https://arxiv.org/pdf/1512.03385.pdf)
 - Data Augmentation
 - Separated to callable functions for easier hyperparameter optimisation, debugging, and more readable code
 - Added custom image input function
 - Added commands while running to eliminate repeated image processing/model training

Model attains ~90% accuracy on validation data and a log loss of ~0.32 on Kaggle's test data.
Results analysed with tensorboard.

Data available at: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition

Modules:
 - Main.py: calls preprocessing.py and CNN_model.py to load and preprocess image data, set model parameters, augment the image collection, and train, test, and produce classification metrics for the model on the validation set.
 - preprocessing.py: Functions for creation of training and test data in .npy files.
 - CNN_model.py: Functions for CNN model creation. Model architecture and more complex hyperparameters can be modified here.
 - modelresults_inspection.py: loads model and test data and outputs images and predicted labels for user inspection.
 - custominput.py: predicts categories and displays images for images located in CustomImageInput directory (will need to be modified to wherever user creates this folder when used).
 - kaggle_submission: loads model to create kaggle submission file (.csv).
