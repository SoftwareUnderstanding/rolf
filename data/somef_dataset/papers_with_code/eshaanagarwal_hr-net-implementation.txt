# hr-net-implementation
My Implementation of Deep High-Resolution Representation Learning for Visual Recognition on Cityscape data with HRnet

Dataset used - (https://www.cityscapes-dataset.com/)
Paper Link - HRNet (https://arxiv.org/pdf/1908.07919.pdf)


![Output Image](output/output1.png)
![Output Image](output/output2.png)

Achieved mIOU 65.22% (without pre-trained backbone)

# ABSTRACT :
Abstract—High-resolution representations are essential for position-sensitive vision problems, such as human pose estimation,
semantic segmentation, and object detection. 

Existing state-of-the-art frameworks first encode the input image as a low-resolution
representation through a subnetwork that is formed by connecting high-to-low resolution convolutions in series (e.g., ResNet,
VGGNet), and then recover the high-resolution representation from the encoded low-resolution representation. Instead, our proposed
network, named as High-Resolution Network (HRNet), maintains high-resolution representations through the whole process. There are
two key characteristics: 
* (i) Connect the high-to-low resolution convolution streams in parallel; 
* (ii) Repeatedly exchange the information across resolutions. 
The benefit is that the resulting representation is semantically richer and spatially more precise. We show the
superiority of the proposed HRNet in a wide range of applications, including human pose estimation, semantic segmentation, and
object detection, suggesting that the HRNet is a stronger backbone for computer vision problems

# APPROACH :
Our network connects high-to-low convolution streams in parallel. It maintains high-resolution
representations through the whole process, and generates
reliable high-resolution representations with strong position
sensitivity through repeatedly fusing the representations
from multi-resolution streams.


# HIGH-RESOLUTION NETWORKS
We input the image into a stem, which consists of two stride2 3 × 3 convolutions decreasing the resolution to 1/4, and subsequently the main body that outputs the representationwith the same resolution ( 1/4).

##  Parallel Multi-Resolution Convolutions
We start from a high-resolution convolution stream as the
first stage, gradually add high-to-low resolution streams
one by one, forming new stages, and connect the multiresolution streams in parallel. As a result, the resolutions for
the parallel streams of a later stage consists of the resolutions
from the previous stage, and an extra lower one.

## Repeated Multi-Resolution Fusions
The goal of the fusion module is to exchange the information across multi-resolution representations. It is repeated
several times (e.g., every 4 residual units).
