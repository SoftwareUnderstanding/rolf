# U-Net implementation for mask prediciton.

This is a solution for Kaggle ship detection competition
[https://www.kaggle.com/c/airbus-ship-detection](https://www.kaggle.com/c/airbus-ship-detection)

It is based on U-Net convolutional network [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)
implemented in PyTorch.

## Installations

This model uses PyTorch 0.4.0, cv2 and skimage for image processing.

## Data

Data contains 192k images of open sea seashore, docks, etc. Around 40k images contain ships, and the rest of the images do not. The aim of the model is to predict is to locate separate ships in the image and find the masks of the ships, i.e. parts of the image where ships are located. The labels for the training set are contained in train_ship_segmentation2.csv (data available at https://www.kaggle.com/c/airbus-ship-detection/data). For every ship in every image the file contains a row with image id and the binary mask for the ship in run-length encoding.


## Files description

The script unet.py does the model construction and training. The script inference.py takes the model constructed in unet.py and performs the prediction. The file nonempty.txt is auxiliary, contains the list of image names that contain ships.

#### Model construction.
Input images are resized to 224x224.The model is constructed as follows: we start with pretrianed resnet34 model available from torchvision. Two fully connected layers are added on top of wtih 2d softmax output. Then the model is trained on the set of all images to predict if an image contains a ship. All the layers of the resnet34 ecept the last one are frozen during the training. Now the U-Net conists of resnet followed by transpose convolution layers:

<div style="width:image width px; font-size:80%; text-align:center;"><img src="https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png" width="70%" alt="alternate text"/> U-Net model architecture, https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/ </div>

Every transpose convolution takes as input one of the intermediate features in the resnet network and the result of the previous transpose convolution step. The output of the 4th layer of resnet model is (n_samples,n_channels,224/23,224/32) Each transpose convolution layer multiplies width and height by 2, and the output of the U-Net has shape (n_samples,224,224), where the value for each pixel in the output is the probability that this pixel is a part of the ship image. The U-Net model is trained only on the set of images containing ships. All layers except three first layers of the resnet model are trained. The loss function used is a combination of [dice loss](https://arxiv.org/pdf/1707.03237.pdf) and [focal loss](https://arxiv.org/pdf/1708.02002.pdf) functions.

#### Model prediction.
Model does prediction in two steps. First the modified resnet network predicts if the image contains ships. If it does, the image is fed to U-Net network that predicts the mask. It outputs one mask per image. This mask is split into connected pieces and the run-length encoded.

## Results
After training for 50 epochs the network has 0.6005 intersection over union score on the validation set (consisting of images with ships only).
For the Kaggle test set the private  leaderbord indicates 0.811146 intersection over union beta score (with winning solution score 0.85448).

Here are some test images and predicted masks:

 <img src="https://raw.githubusercontent.com/neshitov/UNet/master/testimage2.jpg" width="30%" />  <img src="https://raw.githubusercontent.com/neshitov/UNet/master/mask2.png" width="30%" /> 

 <img src="https://raw.githubusercontent.com/neshitov/UNet/master/testimage1.jpg" width="30%" />  <img src="https://raw.githubusercontent.com/neshitov/UNet/master/mask1.png" width="30%" /> 
