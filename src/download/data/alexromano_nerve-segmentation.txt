Competition link: https://www.kaggle.com/c/ultrasound-nerve-segmentation/

The objective of this competition was to segment ultrasound images to find the pixels in each image containing a nerve.
I implemented a U-net style architecture in Keras (https://arxiv.org/abs/1505.04597)

#### Preprocessing:
- Each image was z-scored by subtracting the mean and diving by the std of the training set
- Each mask was normalized in the range 0-1 by dividing by 255

Without augmentation, this model gave accuracy of ~**0.5**

I then added the following data augmentations:
- 30 degree rotations
- vertical and horizontal shifts
- vertical and horiztonal flips
- zooms


Later adding Spatial Dropout between each U-net skip connection was able to boost accuracy from **0.61 to 0.67**
