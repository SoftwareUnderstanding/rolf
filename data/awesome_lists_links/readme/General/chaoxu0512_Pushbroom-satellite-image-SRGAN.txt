# Satellite-image-SRGAN using PyTorch

 Using GAN to do super resolution of satellite images.

<p align="center">
  <img src="http://imghost.cx0512.com/images/2021/02/07/20210207202646.png" title="Fig.1. From left to right: ground truth; srgan result; bicubic result">
</p>

<p align="center"> Fig.1. From left to right: ground truth; srgan result; bicubic result </p>

This is a mini project fulfilled for ELEG5491: Introduction to Deep Learning, The Chinese University of Hong Kong. The course website is [here](http://dl.ee.cuhk.edu.hk/).  

The basic concept comes from the paper  [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802).

## **Research Topic**

Pushbroom satellite image super-resolution using generative adversarial network 

## **Research Background**

Pushbroom imaging [1] has been one of the most common satellite scanning methods over the past decades, which features in low manufacturing cost and relatively high maneuvering.  However, its spatial resolution is usually degraded due to strong atmospheric turbulences, fierce temperature fluctuation or the Earth rotation, etc. As the development of CMOS and CCD technology, the size of single detector pixel is getting smaller and the density of detector array is increasing, which help improve the satellite image resolution but also greatly level up the hardware manufacturing cost. Therefore, software methods such as imaging super-resolution algorithms have been proposed recently.

Since the beginning of this decade, deep learning has been keeping advancing itself into various areas, including object detection, image classification, speech recognition and so on.  Imaging super-resolution has been a hot topic in the deep learning community since it is first introduced in 2014 as SRCNN [2]. In 2017, a perceptual loss is additional proposed in SRGAN [3] as a complement to the conventional pixel-based loss metrics, and it is another landmark in the area of deep learning image super-resolution. Essentially, the SRGAN method is also a variant of GAN proposed in 2014 [4], which is also a great invention.  

## **Research objectives**

I’m going to use GAN-based super-resolution method to help improve the visual effects of the degenerated pushbroom satellite images. The target scale factor is four, which is the super-resolution factor in this project. 

## **Research  Methods**

For the source dataset,  the source image is obtained by a satellite pushbroom scanning method with multiple stitches. Its original size is 1 x 1197 x 50500. We will use the crop-data script to randomly crop the source image and generate the training, validation and testing dataset. In this case, we will generate 96000 training images, 3000 validation images, and 1000 testing images, respectively, with the shape of 1 x 128 x 128. The degraded images will be generated from the downsampling and upsampling of the ground truth.

For the network architecture, it is mainly based on the one proposed in SRGAN. The input channel of feature extractor, generator and discriminator should be accordingly modified to fit the  grey-scale satellite image. Therefore, three experimental groups might be setup to compare the different reconstruction performance. 

For the hyper-parameters, the number of epoch, batch size and decay of learning rate can be  adjusted to investigate different reconstructed outputs. 

| NO.  | Parameter        | Value       | NO.  | Parameter       | Value         |
| ---- | ---------------- | ----------- | ---- | --------------- | ------------- |
| 1    | Operating system | Ubuntu 18.4 | 6    | CPU version     | Intel Xeon(R) |
| 2    | GPU version      | Telsa K80   | 7    | CPU memory      | 47GB          |
| 3    | GPU memory       | 10GB        | 8    | CPU clock speed | 2.5GHz        |
| 4    | GPU number       | 3           | 9    | CPU core number | 12            |
| 5    | CUDA version     | 10.2        | 10   | CPU number      | 4             |

## **Expected results**

Various loss should be calculated and visualized, including real loss, fake loss, discriminator loss, generation loss for batches and epochs.

The trained model should be capable of reconstructing of visual-enhanced images from the degraded images. 

Reconstruction performance for various features might be evaluated, including arc, line, contrast, and so on.

## **Reference**

[1] Mouroulis, P., Green, R. O. & Chrien, T. G. Design of pushbroom imaging spectrometers for optimum recovery of spectroscopic and spatial information. Appl Optics 39, 2210-2220, doi:10.1364/AO.39.002210 (2000).

[2] Dong, C., Loy, C. C., He, K. & Tang, X. Image super-resolution using deep convolutional networks. IEEE transactions on pattern analysis and machine intelligence 38, 295-307 (2015).

[3] Ledig, C. et al. in Proceedings of the IEEE conference on computer vision and pattern recognition.  4681-4690.

[4] Goodfellow, I. J. et al. Generative adversarial networks. arXiv preprint arXiv:1406.2661 (2014).

[5] [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)

  