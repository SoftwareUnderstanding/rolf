# Image-Colourization
Recruitment project for 1st year

## Overview:

  Deep Learning is an upcoming subset of machine learning which makes use of artificial neural
  networks. These are inspired by the human brain and its immense structure and function. Here we
  make use of it to colorize black and white images.
  
  
  ## Methodology:
  
Taking input as a black and white image, this model tries to produce a colorized image. We are
using TensorFlow and Keras API. Our model is trained using Google Colab.

  1. To train the network, we start off with image datasets made up of colourful pictures. The
     datasets used here are CIFAR10 and Landscape Dataset. Then we convert all images from
     the RGB color space to the Lab color space. Just like RGB color space, Lab is an alternate
     color space in which the three
     channels represent:-
     - The L channel represents light intensity only.
     -  The a channel encodes green-red color.
     - The b channel encodes blue-yellow color.

 2. Since the L channel can encode intensity, we use it as the input in our network in the grayscale format. We put all ‘L’ values in an array called ‘X’.
 3. We then use our model to let the network predict the respective a and b channel values. The ‘ab’ values are divided by 128 (feature scaling) so as to reduce values from [-127, 128] to  (-1, 1] to make the learning process quicker and more efficient. These are now put in an array called ’Y’. 
 4. Our model is a Convolutional Neural Network (CNN) which trains on ‘X’ as feature inputs and ‘Y’ as target values.The model consists of a number of convolutional layers. There are no pooling layers in our network. Changes in resolution are achieved using striding and upsampling layers. The kernels are sized 3x3. 
 5. It uses the ADAM optimizer with a learning rate of 0.0003. The loss function used is mean squared error. It also includes a Dropout layer to prevent overfitting. 
 6. 10% of the dataset is used as the validation set. Our network trains for multiple epochs.
 
## Various Models:

### 1. Dog:

<img src="Images/gDog.png" width="700" alt="Graph">
<br><img src="Images/1dog.png" width="500" alt="Output">


### 2. Flower:

<img src="Images/gFlower.png" width="700" alt="Graph">
<br><img src="Images/1flower.png" width="500" alt="Output">


### 3. Landscape:

Trained on Dataset of 3,000 landscape images sized 64x64.
<br><img src="Images/gLandscape.png" width="700" alt="Graph">
<br><img src="Images/1Landscape.png" width="500" alt="Output">


### 4. Landscape 2.0:

Trained with a Dataset of 8,000 landscape images of size 128x128.
<br><img src="Images/gLandscape2.0.png" width="700" alt="Graph">
<br><img src="Images/1Landscape2.0.png" width="500" alt="Output">
<br><img src="Images/2Landscape2.0.png" width="500" alt="Output">
<br><img src="Images/3Landscape2.0.png" width="500" alt="Output">
<br><img src="Images/4Landscape2.0.png" width="500" alt="Output">


### 5. Flower 2.0:

Trained on 3 different datasets to give colourized images of 128x128 pixels. 
<br><img src="Images/gFlower2.0.png" width="700" alt="Graph">
<br><img src="Images/1flower2.0.png" width="500" alt="Output">
<br><img src="Images/2flower2.0.png" width="500" alt="Output">
<br><img src="Images/3flower2.0.png" width="500" alt="Output">
<br><img src="Images/4flower2.0.png" width="500" alt="Output">


### 6. Multiclass:

Trained with 10,000 randomly selected images from CIFAR 100 dataset. CIFAR 100 is a dataset of 50,000 images of size 32x32 belonging to 100 different object classes.
<br><img src="Images/1multiClass.png" width="500" alt="Output">
<br><img src="Images/2multiClass.png" width="500" alt="Output">
<br><img src="Images/3multiClass.png" width="500" alt="Output">

## Source Code:
[Dog](Image_Colourization2(Ship_Airplane_Car_Dog).ipynb)
<br>[Flower](https://colab.research.google.com/drive/18Q7wLrFuXruZ2OulBggtHUlm1dZH8Sw3?usp=sharing)
<br>[Landscape](Image_Colourization(Landscape).ipynb)
<br>[Landscape 2.0](Image_Colourization(Landscape_2_0).ipynb)
<br>[Multiclass](Image_Colourization(Multiclass).ipynb)

## References:
1. Colorful Image Colorization Research Paper by *Richard Zhang, Phillip Isola, Alexei A. Efros*: https://arxiv.org/abs/1603.08511
2. Datasets used:
   -https://www.kaggle.com/arnaud58/landscape-pictures
   -https://www.kaggle.com/huseynguliyev/landscape-classification
   -https://www.cs.toronto.edu/~kriz/cifar.html
3. https://www.pyimagesearch.com/2019/02/25/black-and-white-image-colorization-with-opencv-and-deep-learning/
4. https://www.youtube.com/watch?v=VyWAvY2CF9c
