# Resnet
Residual Network for SIGNS dataset

## Introduciton

This repository contains the residual network model to classify the SIGNS dataset. In recent years, neural networks have become deeper and deeper. They can represent complex functions and learn features at many different levels of abstraction, from edges (at the lower layers) to very complex features (at the deeper layers). However, they have a huge problem: vanishing gradients. To tackle this problem, residual networks have been created.

In ResNets, a "skip connection" allows the gradient to be directly backpropagated to earlier layers:
![](https://github.com/lixihan/hello-world/blob/master/resnet.png "")

The ResNet block with "skip connection" can easily learn an identity function. Adding more ResNet blocks to the middle/end of a big neural network doesn't hurt the performance and doesn't really hurt the neural network compared with the simpler version. 

## Neural Network architecture

There are two different Resnet blocks in this model: the identity block and the convolutional block. The identity block is the standard block in ResNets. The input activation has the same dimension as the output activation. The settings for the convolutional blocks are:

| CONV2D        | Filter Shape  | Stride | Padding | 
| ------------- |:-------------:| -----: | -----:  |
| 1             |          (1,1)|  (1,1) |    Valid|
| 2             |          (f,f)|  (1,1) |     Same|
| 3             |          (1,1)|  (1,1) |    Valid|

The identity block is implemented in identity_block(X, f, filters, stage, block), and the architecture is:

![](https://github.com/lixihan/hello-world/blob/master/Identity_block.png "")

The convolutional block is used when the input and output dimensions don't match. Compared with the identity block, the convolutional block has a convolutional layer in the shortcut path. The input activation has different dimensions from the output activation. The settings for the convolutional blocks are:

| CONV2D        | Filter Shape  | Stride | Padding | 
| ------------- |:-------------:| -----: | -----:  |
| 1             |          (1,1)|  (s,s) |    Valid|
| 2             |          (f,f)|  (1,1) |     Same|
| 3             |          (1,1)|  (1,1) |    Valid|
| shortcut      |          (1,1)|  (s,s) |    Valid|


The convolutional block is implemented in convolutional_block(X, f, filters, stage, block, s = 2), and the architecture is:

![](https://github.com/lixihan/hello-world/blob/master/convolutional_block.png "")

Based on the identity block and the convolutional block, the final architecture is shown as follows:

![](https://github.com/lixihan/hello-world/blob/master/res_model.png "")

The details of the ResNet architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

The paramters are shown as follows: 

Zero-padding:(3,3)

Stage 1: CONV2D has 64 filters of shape (7,7) and uses a stride of (2,2). MaxPooling uses a (3,3) window and a (2,2) stride.

Stage 2: The convolutional block uses 3 set of filters of size [64,64,256], f=3, and s=1. The 2 identity blocks use 3 set of filters of size [64,64,256], and f=3.

Stage 3: The convolutional block uses 3 set of filters of size [128,128,512], f=3, and s=2. The 3 identity blocks use 3 set of filters of size [128,128,512], and f=3.

Stage 4: The convolutional block uses 3 set of filters of size [256, 256, 1024], f=3, and s=2. The 5 identity blocks use 3 set of filters of size [256, 256, 1024], and f=3.

Stage 5: The convolutional block uses 3 set of filters of size [512, 512, 2048], f=3, and s=2. The 2 identity blocks use 3 set of filters of size [512, 512, 2048], and f=3.

2D Average Pooling:(2,2) 

The model is implemented in ResNet50(input_shape = (64, 64, 3), classes = 6) based on Keras.

## SIGNS Dataset
The SIGNS data set is of the shape (64, 64, 3). There are 6 classes, representing number from 0 to 6. Each data point is a gesture picture corresponding to the number.

![](https://github.com/lixihan/hello-world/blob/master/SIGNS_dataset.png "")

After normalization, the training and test labels are transformed into one hot matrix. The details for the dataset are:

| Parameter                  | Value            | 
| -------------              |:-------------:   | 
| number of training examples|              1080| 
|     number of test examples|               120|
|               X_train shape| (1080, 64, 64, 3)|
|               Y_train shape|         (1080, 6)|
|                X_test shape|  (120, 64, 64, 3)|
|                Y_test shape|           (120,6)|


## Run the model
1. Run the model:

model = ResNet50(input_shape = (64, 64, 3), classes = 6)

2. Before training the model, compile the model:

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

We use Adam as the optimizer, the categorical cross-entropy loss as the loss function, and accuracy as the evaluation metric.

3. Fit the model:

model.fit(X_train, Y_train, epochs = 10, batch_size = 32)

We use 10 epochs for fitting. Try other epochs can achieve different results. Better performance can be achieved for ~20 epochs, but this would be time-consuming on a CPU.

## API
The REST API only has one module: API.py, including the following functions:

(1) load_model: load the pre-trained Keras model.

(2) prepare_image: conduct the pre-processing of the image. Obtain the image as the input. Convert the image into the RGB format. Adjust the dimensions of the image to (64, 64, 3). Conduct mean subtraction and feature scaling. 

(3) predict: achieve the API function with the prediction results. 

To set up the environment, you need to satisfy the following requirements:

| Python        |            2.7| 
| ------------- |:-------------:| 
| pillow        |          5.3.0|  
| Flask         |          1.0.2|  
| Tensorflow    |         1.10.0|
| Keras         |          2.2.2|
| numpy         |         1.14.3|


Befofore running the API, you need to set up the environment based on the requirements. To run the requirements:

$ pip install -r requirements.txt

Test the API based on the following steps:

(1) Prepare a sample picture sample.jpg for classfication. 

(2) Run API.py to initialize the REST API service.

$ python API.py

(3) Use curl to request to the /predict endpoint:

$ curl -X POST -F image=@sample.jpg 'http://localhost:5000/predict'

The final classification result will be displayed based on the sample picture. 


## Reference
For more information regarding resnet, please go to https://github.com/KaimingHe/deep-residual-networks and http://arxiv.org/abs/1512.03385.
