# Overview
This is repository is created as a part of final project for Fundamentals of Machine Learning (EEL5840) under Prof Alina Zare in University of Florida for the Master's in Computer Science program. <b>An implentation of Deep convolutional neural network  inspired by the famous "Lenet"(http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) Architecture with Pytorch to recognize Handwritten Characters.</b>

# Requirement
The detailed requirement of this project will be found out in the file <b>project1.pdf</b>

# DataSet
The Dataset is custom handwritten charecters provided by Prof Alina Zare created by her students of Electrical and Computer Engineering Department of the University of Florida. 

# Easy DataSet
The “easy” test set is composed of hand-written ‘a’ and ‘b’ characters. The  code should produce the labels ’1’ for ’a’ and ’2’ for ’b’

# Hard DataSet
The goal is to train the system to distinguish between handwritten characters. The “hard” data set consists of the following characters: ’a’, ’b’,’c’,’d’,’h’, ’i’, ’j’ and ’k’ and ’unknown’. There will be test data points from classes that do not appear in the training data. So, the system have come up with a way to identify when a test point class is “unknown” or what not in the training data. The label you should return for this case is -1.
The code outputs a class label that matches the class value in the provided training data. These should be: 1,2,3,4,5,6,7,8, and -1 respectively.

# Models
We came up with three types of models -
1. with Batch Normalization (https://arxiv.org/pdf/1502.03167.pdf)
2. without Batch Normalization
3. with Dropout (https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)

We first split the dataset into train, test and validation set. Then we used different parameters(learning rates, epochs, batch sizes) to train our network and based on the results we found out the model without batch normalization produces highest accuracy(<b>97.469%</b>). So we selected that model as our final deliverable model.

# Parameters of the Convolution Network
<img src="https://github.com/Shantanu48114860/Handwritten-Character-Recognition/blob/master/Parameters.png" width="500" height="500">

# Crossvalidation framework
  skorch (<b>0.7</b>)
 
# Pytorch version
  <b>1.3.1</b>

# Easy Dataset
  # Training
<b>Training “easy” blind data set</b>
1. For <b>(“easy” blind test data set)</b> all the parameters(ex epoch, learning rate) are listed in <b>./Handwritten-Character-Recognition/train.py</b> file.
2. For specifying the paths for the files of the dataset and label set, please use the variables data_set_path and            label_set_path.
3. Please place the the files of the dataset and label set in the Handwritten-Character-Recognition folder.
4. The model will be generated in the ./Handwritten-Character-Recognition/model folder.
5. All the details of the models during training process will be genrated in the ./Handwritten-Character-Recognition/metrics folder.

# Testing
<b>Testing “easy” blind data set</b>
1. For <b>(“easy” blind test data set)</b> all the parameters(ex epoch, learning rate) are listed in <b>./Handwritten-Character-Recognition/test.py</b> file.
2. For specifying the paths for the files of the dataset, please use the variables data_set_path variable.
  
# Hard Dataset 
 # Training
<b>Training Hard Dataset</b>
1. For <b>(“easy” blind test data set)</b> all the parameters(ex epoch, learning rate) are listed in <b>./Handwritten-Character-Recognition/train_extra_credit.py</b> file.
2. For specifying the paths for the files of the dataset and label set, please use the variables data_set_path and            label_set_path.
3. Please place the the files of the dataset and label set in the Handwritten-Character-Recognition folder.
4. The model will be generated in the ./Handwritten-Character-Recognition/model folder.
5. All the details of the models during training process will be genrated in the ./Handwritten-Character-Recognition/metrics folder.

# Testing
<b>Testing Hard Dataset</b>
1. For <b>(“easy” blind test data set)</b> all the parameters(ex epoch, learning rate) are listed in <b>./Handwritten-Character-Recognition/test_extra_credit.py</b> file.
2. For specifying the paths for the files of the dataset, please use the variables data_set_path variable.
  
# Caution
  To generate a new model with new parameters, please run train.py and train_extra_credit.py file first. Existing models are found at the location <b>Handwritten-Character-Recognition/model</b>
  
# How to run 
# Easy Dataset
1. <b>Training:</b> <br/>
cd Handwritten-Character-Recognition<br/>
<b>python train.py</b>
2. <b>Testing:</b> <br/>
<b>python test.py</b>
  
# Hard Dataset
1. <b>Training:</b> <br/>
cd Handwritten-Character-Recognition<br/>
<b>python train_extra_credit.py</b>
2. <b>Testing:</b> <br/>
<b>python test_extra_credit.py</b>
  
# Final Output
The final output after running the ./Handwritten-Character-Recognition/test.py and ./Handwritten-Character-Recognition/test_extra_credit.py files will be found at the files easy_file.npy and hard_file.npy respectively. In these two files the predicted labels are stored as numpy arrays.

# Project report
Project report is included in the <b>FML_final.pdf </b>

# Final Result on real test data
As per the teaching assistants, when they ran the model on the test data set, the model produces an accuracy of <b>97.3% and 86.5% on the easy and hard test dataset </b> respectively.

