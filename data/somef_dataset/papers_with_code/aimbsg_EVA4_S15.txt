# EVA4_S15
Given an image with foreground objects and background image, predict the depth map as well as a mask for the foreground object

Table of contents :
1) Understanding the problem
2) Applications of depth estimation
3) Data preparation
4) Approach to depth estimation
5) Implementation constraints
6) Result
7) Appendix

# Understanding the problem
Human brain has the remarkable ability to infer depth when viewing a two-dimensional scene (even in a photograph). But accurate depth mapping is a challenge in computer vision which computer vision enthusiasts are trying to solve. The problem I am trying to solve here is to do a monocular depth estimation and object segmentation using custom prepared dataset.

# Applications of depth estimation
Depth information from a scene is invaluable for tasks like augmented reality, robotics, self driving cars etc

# Data preparation
Drive link to dataset : https://drive.google.com/open?id=1Hr1OuftLZ0reDac1yJ2wAspwE9gnUQEi

bg : Forest, park, seashore and road. Total images = 100; Channels = RGB; Size = 905KB; Resolution = 224X224

fg : Men, Women, Children and combination of men-women, women-children and men-children. Total images = 100; Channels = RGB; Size = 576KB; Resolution = 160X160. Gimp is used to remove the background in foreground images(made transparent). Understood difference between white bg and transparent bg.

fg_bg : Randomly placed each fg 40 times(with flips) over each bg. Total images = [100X100X(20X2)] 400K. Channels = RGB; Size = 2.2GB; Resolution = 224X224

fg_bg_mask : fg is converted from RGB to black and overlaid on top of black background. This is done along with step 3 (in the same for loop). Total images = 400K. Size = 1.6GB; Resolution = 224X224

fg_bg_depth : Tweaks with respect to image input folder and save have been made from the shared Dense Depth code. Image loading is done on CPU while prediction is done on GPU. Need to load the data as well in GPU for fast processing. 2000 images takes 15 minutes hence working on optimizations. Could have done this in the same for loop along with steps 3 and 4.

<img src = "Data_Samples_Depth_Model.png">

Link to codes :

Overlap and mask : https://github.com/aimbsg/EVA4_S14/blob/master/EVA4_S14_Overlap_And_Mask.ipynb

Dense depth model : https://github.com/aimbsg/EVA4_S14/blob/master/EVA4_S14_Dense_depth_model.ipynb

# Approach to depth estimation
Create data loader to load the dataset (preferably in batches considering the size of the dataset)  

Use augmentation strategy (resize and normalize using the mean and standard deviation of the dataset)

Create a model which takes fg_bg and bg (stacked over one another as array) as input. This type of stacking does not the change the size of the input while it increases only the number of channels

2 losses to be used,

  i) Comparing the output with fg_bg_mask

  ii)Comparing the output with fg_bg_depth

Run the model for 'n' epochs

Compare train vs validation accuracy. Save the model and change the learning rate and re-run the model for more number of epochs.

# Implementation constraints 
Data I have used to train so far are small (CIFAR10 ~60K, tinyimagenet ~100K) compared to this custom dataset (400K). 

Got time out error while loading the complete dataset in Colab.

As I rely on office laptop for assignment I do not have privilege to use local installation of python and have to use Colab, which I read is the best to solve for large dataset.

So I trained with 100K dataset for 225 epochs.

# Result
Link to code : https://github.com/aimbsg/EVA4_S15/blob/master/EVA4_S15_Custom_Dataset_Depth_Prediction.ipynb

Best mask accuracy : 46.31%

Best depth accuracy : 40.58%

# References
https://arxiv.org/abs/1812.11941

https://arxiv.org/abs/1608.06993

https://towardsdatascience.com/depth-estimation-on-camera-images-using-densenets-ac454caa893

