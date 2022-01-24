# Mini_Project2: A Basic Neural Network Design for Pneumonia Diagnosis   


This program was created to try and classify DICOM chest X-Rays as either Pneumonia positive or negative based on lung opacitites, 
Specifically It builds a simple neural network to try and accomplish the task. The data was taken from Kaggle.com and must be downloaded from https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data. 


## Prerequisites

Required Packages 

```
Python v3.6
Tensorflow v1.5.0
keras 
shuntil 
csv
pydicom 
numpy 
pandas
skimage
itertools
matplotlib
pickle
image
random
PIL

```

## Program Summary 

### 1. pneumonia_posneg.py

The first module exists to preprocess the data and create the training, validation, and testing sets.  The RSNA images are saved in DICOM
medical format and the labels are saves in a .csv file. The data is stored in four directories, stage_1_test_images.zip stage_1_train_images.zip, stage_1_train_labels.csv.zip. The first program 
reads through these directories and begins by creating two dictionaries, on for the testing and validation sets. The dictionaries store a key with the image filename and 
then encodes either a 0(pneumonia negative) or 1(pneumonia_Positive) for the value.  Then it loops throught these dictionaries and converts all the (1024x1024).dcm files to (256x256).png files in order to 
work with the keras flow_from_directory function. specifically the directories much contain the following hierarchy. 

```
data/
    train/
        pneumonia/
            0001.jpg
            0002.jpg
            ...
        no_pneumonia/
            0001.jpg
            0002.jpg
            ...
    validation/
        pneumonia/
            0001.jpg
            0002.jpg
            ...
        no_pneumonia/
            0001.jpg
            0002.jpg
            ...
```
The data was split into 70% training, 30% validation following industry standard guidelines. 
Lastly it creates a test directory and loads all of the test images into it for further possible model verification and testing

### 2. pneumonia_posneg_model.py

The Second module heavily depends on Keras a python wrapper for tensorflow. It first creates the neural network loosely following Lenet5, an existing 
image identification software. Specifically the network includes three convolution and Max Pooling layers are follwed by three dense layers all of which are activated by the Relu function. 
The network concludes with a sigmoid activation to narrow the output down to a single one hot vector signaling either pnemonia positive or negative
It also saves the history of the model to the current working directory for use by the next module 

### 3. pneumonia_posneg_eval.py

The last module first takes the history dictionary returned by the Keras fit_model() function and creates a subplot of the Train/validation accuracy and loss functions for each epoch. Also reloads the model and evaluates some random test images.  Then it plots the test images in quesition, where the plot title is the predicted class. 


## Execution 

Once all the required packages have been installed you are ready to run the program 
1. Download zipped data data files from https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data

2. Unzip the data files with terminal and leave in current directory 

```
$ Unzip stage_1_test_images.zip
$ Unzip	stage_1_train_images.zip
$ Unzip	stage_1_train_labels.csv.zip
```
2. execute the modules in the following order, Make sure to use Python 3

```
$ Python3 pneumonia_posneg.py
$ Python3 pneumonia_posneg_model.py
$ Python3 pneumonia_posneg_eval.py
```
## Discussion and Some Results 
For this mini project I was heavily constrained by HardWare, specifically all models were trained on a 2018 macbook pro intell i7 CPU.  In the future further experimentation and compution should be done via the cloud or more robust hardware. The maximum accuracy I was wable to achieve was 77% after running for 5 epochs. which turned out to be the maximum feasibly possible on my personal computer when using such a large data set.

![alt text](https://raw.githubusercontent.com/astoycos/Mini_Project2/master/LR%20%3D%20.0001%2C%20ss%20%3D%2010000%2C%20ACC%20%3D%20.778.png)

The Figure above shows how the model quickly rises to its maxium accuracy of 77%

![alt text](https://raw.githubusercontent.com/astoycos/Mini_Project2/master/Display_testing_examples.png)

Based on the figure above it seems that diagnosing pneumonia from chest x-rays effetively will require a much larger nerual network architecture along with more imputs, such as bounding boxes, in order to truly be successful. Specifically when we attempt to predict on test images we most often get an output of [0] which leads me to believe that the problem is too complex for such a simple Neural Network. However as demonstration for simple nerual network architecture and data pre-prossing this demonstration was very effective.  

## A Short Neural Net Architecture Comarison: LeNet5 and Resnet 
LeNet was released in 1988 by Yann LeCun and is a pioneering network that paved the way for many of the modern deep learning architectutres used today. It was revolutionary due to the fact that never before had there been a network which used convolutions to extract various image features.  Also, it was built during an era where hardware was a major constrant so being able to tune the various convolutional layers made it efficient enough to run in the pre GPU era.  Specifically the platform was built using three major blocks. First the image is convoluted by various sized filters to extract features, as you go deeper in the network the feature maps change from simply reporesening lines and edges, to being able to recognize macro objects.  Pooling layers follow each convolution and serve to extract the most significant data within a feature map while also decreasing the size of the layer.  Lastly a non-linear activation function is applied, such as a tanh or sigmoid equation. Lastly is a set of dense fully connected layers to serve as a final classifier. Due to the simplistic and tunable nature of this architecture I decided to model the basic pneumonia network following many of the same guidelines. Resnet was released in December of 2015, and is an advance widely used architecture, beating out its predicessor, VGGNet, with and error of only 3.6% in the ImageNet test set. At it's base functional level, ResNet also takes many intuitions from Lenet, sucha as the general order of convolution, pooling and dense layers. However, ResNet is a much deeper network and can be implemented locally with either 50 layers(Resnet50) or 101 layers(Resnet101), while the authors of Resnet have even utilized an implementation with over 1000 layers.  ResNet's ability to utilize such deep network arcitecture without facing a "vanashishing gradient" issue is what allows it to acheive such great results.  Specifically the solution begin with the simple idea of "identity shortcut connection" which is when the output of two convolutional layers along with the bypasesed input in passed to the next consectutive layers. This keeps the backpropagation gradient from steadily going to zero as the algorithm progress though the numerous layers.  


## Authors

* **Andrew Stoycos** 

## Acknowledgments

* Specific code sourses can be found in module source code 
* Genereal code schematic provided by https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
* Data provided by the RSNA and Kaggle.com https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data
* Architecture Comparison sources 
```
https://towardsdatascience.com/neural-network-architectures-156e5bad51ba
https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035
https://arxiv.org/pdf/1801.00631.pdf
https://arxiv.org/abs/1512.03385
http://yann.lecun.com/exdb/lenet/
```


