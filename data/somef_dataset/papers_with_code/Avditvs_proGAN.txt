# proGAN
My own implementation of Progressive Growing of GANs, written in PyTorch. https://arxiv.org/abs/1710.10196

---------

This type of Generative Adversarial Network consists of training the generator and discriminator by gradually inserting new layers at runtime and thus increasing the resolution.  
The loss used in this GAN is the Wasserstein loss with gradient penalty.

## Results on CelebA

These are the results obtained on the celebA dataset. The model was trained during 7 hours on a RTX2060 6GB VRAM.  
The size of the produced images are 128 by 128. These images are not cherry picked

![](imageDataCelebA.png)
