# SuperResolution-using-GANs
pytorch implementation of the paper https://arxiv.org/pdf/1609.04802.pdf

Dataset used was CelebA. which can be obtained from https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8.

162770 images were used for training and the rest were used for testing purposes.

This work was done for my Coputer Vision class for the Spring 2019 semester.
The repo is organised in Codes and Results
There is data pre processing, training and testing code available in the codes folder.

The code in Training.ipynb uses transpose convolution for upsampling.
The code in Training_Extension.ipynb uses pixel shuffler for upsampling.

The results folder has the best trained model which is at epoch 58. It even has the final presentation which has the results for the work. Note that in the presentation the results are shown only for training 15 epochs. The results for more training are present as jpeg files.

