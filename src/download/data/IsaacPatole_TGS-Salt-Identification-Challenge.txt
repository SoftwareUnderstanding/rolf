# TGS-Salt-Identification-Challenge
TGS Salt Identification challenge using U-Net model for Semantic Segmentation using Tensorflow 2.0. 

Overview from Kaggle.com:

'Several areas of Earth with large accumulations of oil and gas also have huge deposits of salt below the surface. But unfortunately, knowing where large salt deposits are precisely is very difficult. Professional seismic imaging still requires expert human interpretation of salt bodies. This leads to very subjective, highly variable renderings.'

I implemented a simple U-Net model so that anybody can understand the architecture in just one glance and can change/modify it if required.

Many important techniques are being used in the model such as Transpose Convolution, adding skip connections, EarlyStopping, reducing the learning using ReduceLROnPlateau etc. Used a binary-corssentropy because we just have to segment if the salt is present or not (0 or 1).

The model achieved 0.0683 loss and  97% accuracy on training data and 0.1458 loss and 90% accuracy on validation data with 50 epochs.

Download the data from the link below.

Reference: 

U-Net: Convolutional Networks for Biomedical Image Segmentation(Paper): https://arxiv.org/abs/1505.04597

U-Net: Convolutional Networks for Biomedical Image Segmentation(Video): https://www.youtube.com/watch?v=81AvQQnpG4Q

Understanding Semantic Segmentation with UNET: https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47

Link to the Dataset on Kaggle.com: https://www.kaggle.com/c/tgs-salt-identification-challenge/

Tutorial on Subplots: https://www.youtube.com/watch?v=afITiFR6vfw&t=486s


Let me know if you need help in understanding U-Net.

