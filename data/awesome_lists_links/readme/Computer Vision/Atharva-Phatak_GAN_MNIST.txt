# PyTorch-MNIST-DCGAN

A PyTorch implementation of Deep convolutional Generative Adversarial Networks(DCGAN) for MNIST dataset.
  
  * You can download 
      * MNIST dataset : <http://yann.lecun.com/exdb/mnist/>

# Results

## MNIST
  * Generated using fixed noise
  
  | DCGAN |
  | ----- |    
  | ![image](https://github.com/Atharva-Phatak/GAN_MNIST/blob/master/MNIST_final/MNIST_gen.gif)|
  
  * MNIST vs Generated Images
  
  | MNIST | DCGAN |
  | ----- |-------|
  |![image](https://github.com/Atharva-Phatak/GAN_MNIST/blob/master/MNIST_final/raw_MNIST.png)| ![image](https://github.com/Atharva-Phatak/GAN_MNIST/blob/master/MNIST_final/MNIST_DCGAN_20.png)|
  
  * Learning Time
    
    MNIST DCGAN : Avg Time per epoch :1691 seconds.If you want to reduce the time change generator(128) to generator(64) and similar for the Discriminator.
 
 # Dependencies
 * PyTorch
 * Python 3.6
 * Numpy
 * Matplotlib
 * Torchvision
 * Imageio
 
 # Reference
 
[1] Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014.

Link : <http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf>

[2] Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).

Link : <https://arxiv.org/pdf/1511.06434.pdf>
