# Using Neural Networks to classify Salt Pixels in a Seismic Image

![siesmic images](https://wholefish.files.wordpress.com/2009/05/salt_domeseismic.jpg)

## README Contents
 - [Introduction](#Abstract)
 - [Methodology](#Methodology)
 - [U-Net](#U-Net)
 - [Model Evaluation](#Model-Evaluation)
 - [Conclusions and Next Steps](#Conclusions-and-Next-Steps)
 - [Sources](#Sources)
 
## Introduction

Salt formations are important in geology because they form one of the most important traps for hydrocarbons.

Interpretation on seismic images has long used texture attributes, to identify and highlight areas of interest. These can be seen like feature maps on the texture of the seismic. The texture of salt in a siesmic image is unique in the output images of collected siesmic data. 

Geologists use 2D or 3D images of seismic images that have been heavily processed to study the subsurface for salt formations. Here I am going to use maching learning and computer vision to identify salt in a seismic image

## Methodology

The problem deals with seismic images and boils down to a multiclass classifcation model. Each pixel in an the siesmic images needs to be classified of being salt or not salt. Then an output image is produced where it can be scored againts the true image.

For this problem I chose to use a U-Net model, which is a model used for semantic segmentation.

## U-Net

![U-NET](http://deeplearning.net/tutorial/_images/unet.jpg)

The U-Net was devoloped by Olaf-Ronneberger for Bio Medical Image Segmentation.The architecture contains two paths. First path is the contraction path (also called as the encoder) which is used to capture the context in the image. The encoder is just a traditional stack of convolutional and max pooling layers. The second path is the symmetric expanding path (also called as the decoder) which is used to enable precise localization using transposed convolutions. Thus it is an end-to-end fully convolutional network (FCN), i.e. it only contains Convolutional layers and does not contain any Dense layer because of which it can accept image of any size.

## Model Evaluation



| Model|Train Score|Test Score|Kaggle Score|
|---|---|---|---|
|U-Net with Resnet| 82.10% | 78.48% | 81.79% |

I am happy with the model scores, but know there is room for improvement. The model scored in the top 30% of the kaggle competition of over 3,500 participants.

## Conclusions and Next Steps

- 1. Try other semantic segmentation models (i.e. R-CNN, FRRN)
- 2. Grid searching over parameters
- 3. More computing power
- 4. Use depth as a parameter
- 5. Different versions of metric IOU (i.e. Dice loss, Lovasz loss)

## Sources 

https://arxiv.org/pdf/1505.04597.pdf

https://github.com/jakeret/tf_unet

https://www.jeremyjordan.me/semantic-segmentation/

https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47

https://www.kaggle.com/pestipeti/explanation-of-scoring-metric

https://medium.com/@arthur_ouaknine/review-of-deep-learning-algorithms-for-image-semantic-segmentation-509a600f7b57

https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c










