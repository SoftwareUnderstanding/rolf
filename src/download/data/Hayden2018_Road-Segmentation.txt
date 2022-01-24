Road Segmentation is a simple Python program using convolutional neural networks to do road segmentation on satellite
image. It is implemented using Pytorch. The U-net architecture used in this project was proposed by : https://arxiv.org/abs/1505.04597<br><br>
The current best weight for the model has around 87% pixel wise labelling accuracy. 
This project is a derived work of my UROP project.<br><br>
The model is trained on 1200 512x512 satellite image with 128x128 binary road or non-road labels. 
The satellite images are downloaded from google map while the labels are generated from corresponding map images.
#
<b>Models.py</b> : This file define the U-net segmentation model of this program. <br>
<b>ModelTrain.py</b> : This script is for training the model. Written for two GPU.<br>
<b>ModelTest.py</b> : This script is for testing the accuracy of the model with GPU.
It should be place in the same directory as Models.py and parameter file.<br>
<b>DataProcessing.py</b> : This file contain functions that are used to prepare data for training.<br>
<b>CovNet_param</b> : Parameters of the model.
