# CSCI5922_Project
Neural Networks and Deep Learning Course Project - CSCI 5922

### Project Title: Deep Learning for Cell Segmentation in Time-lapse Microscopy
##### Team Members: Shemal Lalaji, Swathi Upadhyaya, Yuvraj Arora 

#### Objective: 

The goal of our project was to build Neural Network Models to segment moving cells in a real 2D time-lapse microscopy videos of cells along with computer generated 2D video sequences simulating whole cells moving in realistic environments. The evaluation method used for the models is segmentation accuracy.

#### Dataset:

From the vast dataset available in the Cell Tracking Challenge, we chose the Fluo-N2DH-SIM+ dataset. This dataset consists of simulated nuclei of HL60 cells stained with Hoescht. The video is recorded over 29 minutes to study the cell dynamics of various cells. The benchmark for the segmentation evaluation methodology is 80.7 % for this dataset.


#### Model Architecture:

##### 1. U-Net:

![Fig.1 U-Net Model](https://github.com/yuvrajarora/CSCI5922_Project/blob/master/Assets/Unet-Model.png)

U-Net is built on Fully Convolutional Network. It is modified and extended in a way such that it works with very few training images and yields more precise segmentation. The network aims to classify each pixel. This network takes a raw input image and outputs a segmentation mask. A class label is assigned to each pixel. This architecture consists of two main parts: Contraction Path and Expansion Path. We end up creating multiple feature maps and the network is able to learn complex patterns with these feature maps. The Contraction path helps to localize high resolution features and the Expansion Path increases the resolution of the output by upsampling and combining features from the contraction path.


##### 2. U-Net with Convolution LSTM Block - 

![Fig.2 U-Net C-LSTM Model](https://github.com/yuvrajarora/CSCI5922_Project/blob/master/Assets/Unet-CLSTM.png)

This network incorporates Convolution LSTM (C-LSTM) Block into the U-Net architecture. This network allows considering past cell appearances at multiple scales by holding their compact representations in the C-LSTM memory units. Applying the CLSTM on multiple scales is essential for cell microscopy sequences since the frame to frame differences might be at different scales, depending on cells' dynamics. The network is fully convolutional and, therefore, can be used with any image size during both training and testing.

##### 3. VGG Net with Skip - 

![Fig.3 U-Net C-LSTM Model](https://github.com/yuvrajarora/CSCI5922_Project/blob/master/Assets/VGG-Net.png)

VGG Net shows a improvement on the classification accuracy and generalization capability on our model. Along with that using skip and Relu  allows us to improve the performance of the models and segment cells properly to view and  refine the spatial precision of the output.

VGGnet developed by visual geometry group with skip connections consists of 16 conv layers and 1 skip connection. VGG-16 consists of three additional 1x1 conv layers which helps us in out problem to clearly see the segmentation and masking.
It projects the images in similar higher dimensionality.
Along with that a skip connection from the first layer to the layer where the convolution has finished and before usage of skip connection improves the accuracy and helps to see the information more clearly.

We make use of the skip as for our problem we are doing downsampling and then upsampling later but the input to the upsample is a lower resolution picture.
The spatial precision is lost during the downsampling and hence to compensate the resolution loss a skip architecture combines the coarse layer with the shallow layer to refine the spatial precision of the output.

Architecture:
The architecture consists of the following:
1. 2 Conv2d with 64 filters
2. 2 Conv2d with 128 filters
3. 3 Conv2d with 256 filters
4. 3 Conv2d with 512 filters
5. 2 Conv2d with 4096 filters
6. 4 MaxPool with stride as (2,2)
7. Densing to 1x1 with sigmoid
8. 1 skip net from output to input
9. Conv2d Transpose for upsampling
10. Leaky Relu

Note:

The other things that we tried were:

1. Bidirectional LSTM with convolution-
Mostly used for sequences. In this we go two ways from 0 to N and from N to 0 and combine the outputs via summation. Exciting work, but ran into problems and all problems of these were on sequences. 

#### Evaluation Metrics:
Jaccard Similarity Index: 
![](https://github.com/yuvrajarora/CSCI5922_Project/blob/master/Assets/Eval_Metric.png)

where, 
                  R : Pixels belonging to reference object
                  S : Pixels belonging to segmented object


#### Results:

##### Original Image

![Fig.4 Original Image](https://github.com/yuvrajarora/CSCI5922_Project/blob/master/Assets/Original_Img.png)

##### Masks generated:

###### U-Net mask
![](https://github.com/yuvrajarora/CSCI5922_Project/blob/master/Assets/U-Net_Mask.png)
###### U-Net CSLTM mask
![](https://github.com/yuvrajarora/CSCI5922_Project/blob/master/Assets/ConvLSTM_Mask.png)
###### VGG Net mask
![](https://github.com/yuvrajarora/CSCI5922_Project/blob/master/Assets/VGG-Net_Mask.png)

##### Hyperparameters for the model 
![](https://github.com/yuvrajarora/CSCI5922_Project/blob/master/Assets/Result_Table.png)

#### References:

1. Ulman, Vladimír & Maška, Martin (2017). An Objective Comparison of Cell Tracking Algorithms. Nature Methods. 14. 10.1038/nmeth.4473
2. O. Ronneberger, P. Fischer, T. Brox, U-net: Convolutional networks for biomedical image segmentation, 2015.
3. A fully convolutional network for weed mapping of unmanned aerial vehicle (UAV) imagery, Huasheng Huang, Jizhong Deng, Yubin Lan , Aqing Yang, Xiaoling Deng, Lei Zhang
4. [Learning how to train U-Net model by Sukriti Paul](https://medium.com/coinmonks/learn-how-to-train-u-net-on-your-dataset-8e3f89fbd623)
5. [U-Net by Zhixuhao](https://github.com/zhixuhao/unet)
6. [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://papers.nips.cc/paper/5955-convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting.pdf)
7. https://arxiv.org/abs/1409.1556
8. https://www.quora.com/What-is-the-VGG-neural-network
9. https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
10. https://github.com/aaronphilip/Image-Segmentation-On-Faces
