# Image Colorization using CNN
This repository contains a image colorization system using Convolutional Neural nets. The fundamental idea is to predict A and B channels of LAB space images provided the L channels. CIE L*a*b* or sometimes abbreviated as simply "Lab" color space expresses color as three numerical values, L* for the lightness and a* and b* for the green–red and blue–yellow color components. For further details of the color space kindly refer to the following link:

https://en.wikipedia.org/wiki/CIELAB_color_space

The architechture of the network is given by the following

![alt text](https://github.com/Arghyadeep/Image-Colorization-using-CNN/blob/master/process.png)

The idea is inspired by Richard Zhang's image colorization paper: https://arxiv.org/pdf/1603.08511
But instead of upsampling and finding A and B values of the predicted channels from a probability distribution of 313 values as mentioned in the paper, a simple square loss is used to predict. Though simple to implement, a downside of this Loss function is that the images lose its vibrancy in many cases. 

## Process

Around 200000 images from imagenet were used to train. Images used were of resized to 32*32, 64*64 and 128*128.
Initially the models were built using tensorflow. Further details can be found here: https://github.com/Arghyadeep/Image-Colorization-using-CNN/blob/master/report/final%20report.pdf
Next 128*128 sized images were trained using Keras, which made things simpler. The network had a tendency to quickly overfit and hence hyperparamter tuning was essential using a validation set. The hyperparamters tuned were:

1. Kernel size
2. Number of filters
3. Dropout
4. Epochs (For early stopping)
5. Batch Size
6. Stride

## Results

The following are the results on 32*32 images (Left: Grayscale, Middle: Ground Truth, Right: Predicted image)

![alt text](https://github.com/Arghyadeep/Image-Colorization-using-CNN/blob/master/result32.png)

The following are the results on 64*64 images (Right: Ground truth, Left: Predicted image)

![alt text](https://github.com/Arghyadeep/Image-Colorization-using-CNN/blob/master/result64.png)






