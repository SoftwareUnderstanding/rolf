# CMR image segmentation

Pytorch implementation of 2D U-Net for image segmentation based on [*U-Net: Convolutional Networks for Biomedical Image Segmentation*](https://arxiv.org/abs/1505.04597) .
To run the code, download the images from (https://www.ub.edu/mnms/) and put them in ```labelled/```.
The figures below show some segmentation examples and a boxplot of the dice score. The model achieved a mean (standard deviation) dice score of 0.84 (0.28).
![Segmentation examples](pic_segmentation.png)
![Boxplot dice score](boxplot_dice.png)
