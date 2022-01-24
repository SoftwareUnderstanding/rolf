# UNet-Image-segmentation

U-Net is a convolutional neural network that was developed for biomedical image segmentation at the Computer Science Department of the University of Freiburg, Germany. The network is based on the fully convolutional network and its architecture was modified and extended to work with fewer training images and to yield more precise segmentations.


The architecture contains two paths. First path is the contraction path (also called as the encoder) which is used to capture the context in the image. The encoder is just a traditional stack of convolutional and max pooling layers. The second path is the symmetric expanding path (also called as the decoder) which is used to enable precise localization using transposed convolutions. Thus it is an end-to-end fully convolutional network (FCN), i.e. it only contains Convolutional layers and does not contain any Dense layer because of which it can accept image of any size.

## Architecture of UNet

![Unet Architecture](/unet.png)


## About DataSet

The Dataset is taken from a kaggle challenge. <b>We are just going to use training data for demonstration of UNet</b>

The data is a set of images chosen at various locations chosen at random in the subsurface. The images are 101 x 101 pixels and each pixel is classified as either salt or sediment. In addition to the seismic images, the depth of the imaged location is provided for each image. The goal of the competition is to segment regions that contain salt.

## Implementation
Implementation of UNet can be found in Notebook in the repository. It is done in `Pytorch` with `jupyter lab` environment.

# References
- https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47
- https://arxiv.org/abs/1505.04597
- https://en.wikipedia.org/wiki/U-Net
