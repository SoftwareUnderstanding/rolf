# Team-Challenge
Team challenge group 1
Written by: Colin Nieuwlaat, Jelle van Kerkvoorde, Mandy de Graaf, Megan Schuurmans & Inge van der Schagt

# General description:
    This program performs segmentation of the left vertricle, myocardium, right ventricle
    and backgound of Cardiovascular Magnetic Resonance Images, with use of a convolutional neural network based on the well-known U-Net     architecture, as described by [https://arxiv.org/pdf/1505.04597.pdf](Ronneberger et al.) For each patient, both a 3D end systolic       image and a 3D end diastolic image with its corresponding ground truth segmentation of the left ventricle, myocardium and right         ventricle is available. 
    
    The available code first divides the patients data into a training set and a test set. The training data is then loaded from the        stored location and subsequently preprocessed. Preprocessing steps include resampling the image to the same voxel spacing, removal of outliers, normalization, cropping and one-hot encoding of the labels. Before training, the trainingset is subdivided again for training and validation of the model.
    
    For training, a network based on the U-Net architecture is used and implemented with keras. For training, many different variables       can be tweaked, which are described in some detail below. After training, the network is evaluated using the test dataset. This data is loaded and preprocessed in the same way as the training dataset and propagated through the network to obtain pixel-wise predictions for each class. These predictions are probabilities and are thresholded to obtain a binary segmentation. 
    
    The binary segmentations are then evaluated by computing the (multiclass) softdice coefficient and the Hausdorff distance between the obtained segmentations and the ground truth segmentations. The softdice coefficients and Hausdorff distances are computed for each image for each individual class and the multiclass softdice for all the classes together. These results are all automatically saved in a text file. Furthermore, the obtained segmentations as an overlay with the original images, the training log and corresponding plots and the model summary are also saved automatically.
   
    Lastly, from the segmentations of the left ventricular cavity during the end systole and end diastole, the ejection fraction is calculated. This value is, alongside the ejection fraction computed from the ground truth segmentations, stored in the same text file with results.
   

# Contents program:
    - TC_main.py:  Current python file, run this file to run the program.
    - TC_model.py: Contains functions that initializes the network.
    - TC_data.py:  Contains functions that initializes the data, preprocessing and 
                   metrics used in training, evaluation & testing.
    - TC_test.py:  Contains functions that show results of testing. 
    - TC_visualization.py: visualises the intermediated and final results.
    - TC_helper_functions.py: contains functions to make the main more clean
    - Data: a map with all the patient data.
    
    
# Variables:
    
    trainnetwork:       Can be set to True or False. When set to True, the network
                        is trained. When set to False, a network is loaded from the
                        networkpath.
    evaluatenetwork:    Can be set to True or False. When set to True, the network is
                        evaluated. If set to False, no evaluation is performed
    networkpath:        Path to a stored network 
    trainingsetsize:    Number between 0 and 1 which defines the fraction of the data
                        that is used for training. The rest of the data will be used for testing.
    validationsetsize:  Number between 0 and 1 which defines the fraction of the 
                        training set that will be used for validation. The rest of the data will be used for training.
    cropdims:           Dimensions that the cropped images should have.
                        
    num_epochs:         Integer that defines the number of training iterations.
    batchsize:          The number of samples that will be used in one pass through the network.
       

    dropout:            Can be set to True or False in order to activate dropout layers
                        in the Network.
    dropoutpct:         Float between 0 and 1 which defines the amount of dropout
                        each dropout layer uses.   
    activation:         Activation function, can be set to a string to define the activation function used after each 
                        convolution layer (e.g. 'relu')

    lr:                 Float (>=0) which defines the initial learning rate for the stochastic gradient descent (SGD) optimization                               algorithm.
    momentum:           Float (>=0) which defines the amount of momentum used for the SGD algorithm.
    decay:              Float (>=0) which describes the decay of the learning rate after each epoch
    nesterov:           Whether to apply Nesterov momentum. Can be set to True or False.
    
    lr_schedule:        Can be set to True or False. When set to True, a custom learning rate schedule will be used as described by
                        the scheduler function in TC_helper_functions.py.
    
    
    
# Python external modules installed (at least version):
    - glob2 0.6
    - numpy 1.15.4
    - matplotlib 3.0.1
    - keras 2.2.4
    - SimpleITK 1.2.0
    - scipy 1.1.0

# How to run:
    Place all the files of zip file in the same folder together with the Data folder. 
    Make sure all modules from above are installed.
    Run TC_main.py in a python compatible IDE.
    If you want to train your network, set trainnetwork to True in main()
    If you want to evaluate your network, set evaluationnetwork to True in main() and change networkpath to the network you want to evaluate
    (you can find these at global settings).

