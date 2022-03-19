# Wespe-Keras


This repository contains an unofficial implementation of WESPE paper in Keras. There are some modifications such as the use of the Identity loss which is not used in WESPE but used in CycleGAN and the use of InstaceNormalisation layer which improved the stability of the training.

This repository is an unofficial implementation of the WESPE GAN in Keras (https://arxiv.org/pdf/1709.01118.pdf). The paper achieves unsupervised/weakly supervised smartphone image enhancement by mapping images from the domain of phone images to the domain of DSLR images (denoted as domain A and B respectively) using an architecture inspired by the CycleGAN (https://arxiv.org/pdf/1703.10593.pdf). The architecture of Wespe is shown below.

<p align="center"> 
<img src="https://github.com/GBATZOLIS/Wespe-Keras/blob/master/preliminary%20results/main_architecture.png">
</p>



The training image x is mapped from domain A --> domain B using the Forward Generator G. The image G(x) is input to two discriminators D<sub>c</sub> and D<sub>t</sub> (D<sub>c</sub> decides whether the image is a real domain B or an enhanced domain A based on its color distribution, while D<sub>t</sub> decides based on its texture). Finally, the generated image G(x) is mapped back to domain A by the backward generator F. 4 different losses are used: 2 Adversarial (L<sub>color</sub> and L<sub>texture</sub>), a total variation (L<sub>tv</sub>) loss on the enhanced image G(x) and a cycle-consistency loss (L<sub>content</sub>)  on the reconstructed image F(G(x)) (some norm of x-F(G(x) is minimised, the paper defines the content loss on the VGG19 feature space). 

I have modified the model proposed by the paper because some crucial training details were not provided which made it very difficult to find the right combination of all training parameters for stable GAN training. The **main modifications** are:

* **G generator has greater capacity than the backward Generator F**. My intuition for this change was the fact that G learns a more complex mapping (LR --> HR), while F learns a less complex mapping (HR --> LR).

* **Different Discriminator architecture**. I have used the PatchGAN discriminator used in the CycleGAN and CINCGAN models. The difference between a PatchGAN and regular GAN discriminator is that the regular discriminator maps a MxM image to a single scalar output, which signifies "real" or "fake", whereas the PatchGAN maps an M x M image to an N x N array of scalar outputs X<sub>ij</sub>, where each X<sub>ij</sub> signifies whether the patch <sub>ij</sub> in the image is real or fake. What is the patch <sub>ij</sub> in the input image? The output X<sub>ij</sub> is just a neuron in a CNN, thus we can trace back its receptive field to find the input image pixels that it is sensitive to. In the CycleGAN architecture, the receptive fields of the discriminator turn out to be 70x70 overlapping patches in the input image. In our case, the receptive field of each overlapping patch is smaller. My intuitive explanation of why this approach performs better is that there are regions in the enhanced image G(x) which are closer to target domain statistics than other regions of the image. Therefore, the fact that the PatchGAN classifies many overlapping patches of the image as real or fake gives more feedback to the Generator.

* **A cycle reconstruction loss in both domain A and B**. I have discovered that imposing a cycle reconstruction loss in both domain A and B significantly improved the performance of the network compared to using a cycle reconstruction loss only in domain A.


Image enhancement is achieved by mapping images from the domain of phone images to the domain of DSLR images (denoted as domain A and B respectively in the code).

## Getting Started


Steps to run the training:

* Put the training and test data of domains A and B under the folders data/trainA, data/trainB, data/testA and data/testB

* run the model.py file (you can change the patch size, epochs, batch_size and other parameters in the main)

* run the modelwithVGGloss.py file (I have tuned the hyperparameters based on preliminary testing on the DPED dataset. You will probably have to tune the hyperparameters of the model for different domain A and B datasets)


## Requirements

* keras (tensorflow backend)
* scipy
* Pillow
* scikit-image


## Preliminary experiments/results

Visual results after 1 and 2 epochs (about 1.5h of training time in GTX 2080-ti) are saved in the folder "sample images"

Qualitative & quantitative results of the full training and the trained model will be released soon

The model was trained for 7 epochs on 1.5% of the training DPED data.

The evolution of the average SSIM value on the test data of the DPED dataset:

<p align="center"> 
<img src="https://github.com/GBATZOLIS/Wespe-Keras/blob/master/preliminary%20results/ssim_curve.png">
</p>

Visual results after 4 epochs.
![Image 14](https://github.com/GBATZOLIS/Wespe-Keras/blob/master/preliminary%20results/Figure_14.png)
![Image 27](https://github.com/GBATZOLIS/Wespe-Keras/blob/master/preliminary%20results/Figure_27.png)
![Image 13](https://github.com/GBATZOLIS/Wespe-Keras/blob/master/preliminary%20results/Figure_13.png)
![Image 1](https://github.com/GBATZOLIS/Wespe-Keras/blob/master/preliminary%20results/Figure_1.png)
![Image 20](https://github.com/GBATZOLIS/Wespe-Keras/blob/master/preliminary%20results/Figure_20.png)
![Image 26](https://github.com/GBATZOLIS/Wespe-Keras/blob/master/preliminary%20results/Figure_26.png)
![Image 16](https://github.com/GBATZOLIS/Wespe-Keras/blob/master/preliminary%20results/Figure_16.png)
![Image 15](https://github.com/GBATZOLIS/Wespe-Keras/blob/master/preliminary%20results/Figure_15.png)
![Image 22](https://github.com/GBATZOLIS/Wespe-Keras/blob/master/preliminary%20results/Figure_22.png)
![Image 23](https://github.com/GBATZOLIS/Wespe-Keras/blob/master/preliminary%20results/Figure_23.png)
![Image 5](https://github.com/GBATZOLIS/Wespe-Keras/blob/master/preliminary%20results/Figure_5.png)

