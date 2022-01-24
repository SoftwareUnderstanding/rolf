# Pose-Estimation-with-Stacked-Hourglass
An implementation of a state of the art pose estimation model in tensorflow 2.0.

The model in question is the **Stacked Hourglass Network for Human Pose Estimation**, see :

https://arxiv.org/abs/1603.06937

The official implmentation was originally made in Pytorch, see :

https://github.com/princeton-vl/pose-hg-train

## train.py

The model is trained on the MPII Human Pose Image dataset :

http://human-pose.mpi-inf.mpg.de/#download

The training and evaluation was not fully tested because of my laptops performance and the size of the dataset (~12Gb), making it difficult to upload for training in Colab.

In the following I used a pre-trained model using the same parameters as the official PyTorch implementation.

```
Optimizer = RMSprop
learning rate = 5e-4
loss = MSE
epochs = 96
batch_size = 24
num_stack = 2
```

## main.py

Runs an inference on an image and draws detected limbs and articulations on input image, download pre-trained models from here :

https://drive.google.com/drive/folders/11UB0KZqPJINe6se8kg_2h1QWWxcfcAHW?usp=sharing

The following results are obtained with a threshold value of 0.1.

![](output/images/ronaldo.jpg)

![](output/images/yoga.jpg)

## Remark

Since the following network was trained on an image dataset, there is a lack of consistency when doing inference on succesive video frames.

Henceforth arises a need for utilizing tracking and localization methods such as optical flow and Kalman Filters.
