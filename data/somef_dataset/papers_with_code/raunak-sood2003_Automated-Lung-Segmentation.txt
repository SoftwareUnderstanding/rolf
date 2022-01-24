# Automated-Lung-Segmentation
## Overview
This repository contains the code for lung segmentation using the VESSEL-12 data set (https://vessel12.grand-challenge.org/), which contains 20 lung CT volumes each consisting of 300-500 slices. Segmentation is used to remove unnecessary portions of the CT, leaving only the lung area in the image. My goal was to comapre various techniques for autonomous segmentation of lung CT, including vanilla UNETs trained on varying loss functions, UNETs with pretrained encoders and conditional generative adversarial networks.
## Model Development
The UNET model was adapted from the original paper (https://arxiv.org/abs/1505.04597) in which the input image size was 572 x 572 and the output mask was 388 x 388. 
<p align="center">
  <img src="https://github.com/raunak-sood2003/Automated-Lung-Segmentation/blob/master/Images/unet_unet15.png" />
</p>
In this implementation, the input size was altered to account for standard DICOM image size (512 x 512), and the output mask shape matched the input size by adding a padding of one to each convolutional layer. The UNET model is split into two parts: the encoder and decoder. In the encoding phase, convolutional layers alternate with max pooling layers to downsample the image as the number of channels increase. Then in the decoding phase, transpose convolutional layers are used to upsample the encoded tensor. The grey arrows in the image represent concatenations that assist with upsampling.

## Training
The UNET model was trained on three different loss functions: binary crossentropy (BCE), mean squared error (MSE) and soft dice loss. Cross entropy and dice loss are the traditional losss functions used in segmentation tasks, although MSE has shown promising results in some studies. Additionally, the models were trained for 15 and 30 epoch with a batch size of 5 and a learning rate of 0.01. Google Cloud Platform was used for training the models. Models were trained using a 16 GB Nvidia Tesla K80 GPU.

<p align="center">
<img src="https://github.com/raunak-sood2003/Automated-Lung-Segmentation/blob/master/Images/loss_unet15.png" width="500"/>
  <img src="https://github.com/raunak-sood2003/Automated-Lung-Segmentation/blob/master/Images/ious_unet15.png" width="500"/>
</p>

|                   | Train IOU |  Val IOU  |  Test IOU   |Train Loss |  Val Loss |  Test Loss   |
| :-----            | :---------| ---------:|------------:|----------:|----------:|-------------:|
|BCE_UNET_15_epochs | 0.99225839| 0.97326782|0.9853959925 | 0.00563051| 0.01408213|0.01404043255 |
|MSE_UNET_15_epochs |0.99110307 | 0.9800711 | 0.9924166463|0.0019727  | 0.00214739|0.002164409328|
|DICE_UNET_15_epochs| 0.99095268| 0.98127504|0.9930113867 |0.00604069 | 0.00506628|0.005660841359|

The models were compared based on loss as well as intersection over union (IOU). IOU is calculated by dividing the number of pixels that overlap in the predicted mask and ground truth by the total number of pixels. Based on the table, the model trained using soft dice loss produced the best IOU score on the testing set.
## Example Predictions
<p align="center">
  <img src="https://github.com/raunak-sood2003/Automated-Lung-Segmentation/blob/master/Images/lung_mask_pred_unet15.png" />
</p>
