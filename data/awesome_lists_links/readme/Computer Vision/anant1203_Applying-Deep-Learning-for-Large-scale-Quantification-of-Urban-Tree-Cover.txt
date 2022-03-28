# Applying-Deep-Learning-for-Large-scale-Quantification-of-Urban-Tree-Cover
To find out vegetation cover using deep learning model that can be deployed on the edge device. Dataset used to train the model is cityscape dataset. Model used are Unet and Mobile net V2 model.

# Introduction:
Recent advancement in deep learning has become one of the most powerful tools to solve the
image classification and segmentation problem.Deep learning model learn the filter that helps
in extraction and learning of the important feature form the images. These feature helps to find
differences as well as similarities amongst the image. Deep learning models require large
dataset to learn the complex data representation.In the paper[1] the authors have used DCNN
model to find the green cover in the cities using Cityscapes dataset. Cityscapes dataset has
2975 images and mask of green cover of different cities around the world which was used as
the training data and 500 image with masks were used as testing dataset. The images were
google street view images. The DCNN model has an IOU of 61.2 percent. In this approach I
used state of the art unet model and mobile net v2 model. Unet gave an IOU of 74.5 percent
and mobile net v2 model gave an IOU of 64.3 percent which were better and lighter model than
previously used DCNN model. The model were even tested on different machine type with
different configuration to check their performance.

# Methodology:
### A. Model:
Unet[2] is a state of the art image segmentation model,the architecture looks like a U. It has 3
sections:

● Contraction: It has 3x3 convolutional layer followed by 2x2 max pooling layer
● Bottleneck: It mediate between contraction and expansion.It uses two 3X3 CNN layers
followed by 2X2 up convolution layer.
● Expansion: It uses 3x3 convolution layer followed by 2x2 layer transposed convolution
for upsampling.

Mobile Net V2[3] is small size model made by google to use in mobile/edge devices. It is
classification model which was tweak and can be used for segmentation purpose as well.
The tweaked mobile net model uses Unet architecture in which contraction layer was replaced
by the mobile net v2 model while the upsampling layer remained the same. Mobile net is small
and has smaller complexity as it makes use of the depth wise convolution followed by point wise
convolution instead of normal convolution[4].

|               | Number of parameters | Size of the model | Training and Calibration               |
|---------------|----------------------|-------------------|----------------------------------------|
| Mobile net v2 | 6,504,227            | 9.79 MB           | Pre-trained on full Cityscapes dataset |
| Unet          | 33,480,577           | 50.25 MB          | Pre-trained on full Cityscapes dataset |

Table 1: Show comparison of Unet and mobilenet  model

### B. Dataset:
Cityscapes dataset is used which has a total of 2975 image and mask as shown in figure 1.

![](https://github.com/anant1203/Applying-Deep-Learning-for-Large-scale-Quantification-of-Urban-Tree-Cover/blob/master/image/zurich_000121_000019_gtFine_color.png)
![](https://github.com/anant1203/Applying-Deep-Learning-for-Large-scale-Quantification-of-Urban-Tree-Cover/blob/master/image/zurich_000121_000019_leftImg8bit.png)
Figure 1: Examples of mask and image used to train the model.

The images and mask had 1024x 2048 dimensions ​ which was brought down to 512x512
dimension. As we can see that the mask had multiple class but since this problem deals with
vegetation cover so we converted the mask to black and white with vegetation cover as white
and rest as black. Json file provided with dataset, having coordinates of different
classes(objects), was used to convert mask as per our usage as shown in figure 2.

![](https://github.com/anant1203/Applying-Deep-Learning-for-Large-scale-Quantification-of-Urban-Tree-Cover/blob/master/image/2.png)
<br>
Figure 2: Example of mask with vegetation cover and image with size 512x512.

### C. Evaluation Matrix:
Mean IoU was used for measuring the accuracy of the location of labelled vegetation labels.
<br> n = number of images in test set
<br> TP = true positive predicted vegetation labels for image i
<br> FP = false positive predicted vegetation labels for image i
<br> FN = false negative predicted vegetation labels for image i
<br> IoU = TP/( TP + FP + F N)
<br> Mean IoU = (1/n) * Summation(IoU)

# Results:
### A. Model Performance:
Both the model outperform the DCNN model mentioned in [1]. The number of parameters in the
Unet model was almost half of what was mentioned in DCNN but still it was able to outperform it
with the IoU of **74.5** percent. The mobile net v2 model had almost one tenth of the number of
parameters as in DCNN model but still it out perform it with the IoU of **64.3**. Figure 3 shows the
result of both the models.
<br>
![](https://github.com/anant1203/Applying-Deep-Learning-for-Large-scale-Quantification-of-Urban-Tree-Cover/blob/master/image/1.png)

### Performance Testing:
The model were tested on different environment to check the scalability of the model. The result
can be seen in table 2.

| Number of Parameter in mobile net = 6,504,227        Size: 9.79 MB <br> Number of Parameter in Unet = 33,480,577            Size: 50.25 MB |
|---------------------------------------------------------------------------------------------------------------------------------------|

| Machine Type        | GPUs                | UNet        | Mobile Net V2 | Number of image |
|---------------------|---------------------|-------------|---------------|-----------------|
| 8 CPU, 52 GB RAM    | NVIDIA Tesla T4 x 1 | 85 sec      | 17 sec        | 500             |
| 4 CPU, 15 GB RAM    | None                | 3500 sec    | 338 sec       | 500             |
| 2 vCPUs, 13 GB RAM  | None                | 6500 sec    | 591 sec       | 500             |
| 2 vCPUs, 7.5 GB RAM | None                | 120 sec     | 16 sec        | 10              |
| 1 vCPU, 3.75 GB RAM | None                | Did not run | 5 sec         | 1               |
| 1 vCPU, 3.75 GB RAM | None                | Did not run | 6 sec         | 2               |

Table 2: Show Unet and Mobile net performance on different environment. 

Google cloud platform was used to train and test the model. The training required 8 CPU, 52 GB
RAM and NVIDIA Tesla T4 x 1. The Unet model took almost 5 hrs to train 50 epoch with batch
size of 1. While mobile net v2 only took 2 hrs to train 200 epoch with variable batch size. Both
models were trained on 2975 images.

# References:
<br>[1] http://senseable.mit.edu/papers/pdf/20180920_Cai-etal_Treepedia-2_IEEE-Conference.pdf
<br>[2] https://arxiv.org/pdf/1505.04597.pdf
<br>[3] https://arxiv.org/pdf/1704.04861.pdf
<br>[4] https://towardsdatascience.com/review-mobilenetv1-depthwise-separable-convolution-light-Weight-model-a382df364b69



