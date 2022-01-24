# GAN_Advance-Machine-Learning
Final project for GR5242_Advanced Machine Learning

Protential reference:

https://blog.csdn.net/qq_33594380/article/details/84135797    --DCGAN

https://www.cnblogs.com/bonelee/p/9166122.html   --DCGAN、WGAN、WGAN-GP、LSGAN、BEGAN

https://github.com/rajathkmp/DCGAN   -- Code of DCGAN using keras (debug failed)

https://www.tensorflow.org/tutorials/generative/dcgan   -- Code of Basic DCGAN (debug success)


Implement of GAN:

From the paper. https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf, under-
stand core ideas of GAN. Make sure to understand Figure 1 and Algorithm 1 of the paper. Also, why we
want G to minimize and D to maximize V (G;D).
1. Implement your own GAN with CNN layers on MNIST data. Please describe the architectures of your
generator and discriminator and also any hyper-parameters chosen. Post a plot of training process from
tensorboard to make sure that the networks are trained as expected. To help guide you, an example of
GAN on MNIST can be found in https://www.tensorflow.org/tutorials/generative/dcgan, but
importantly, you must develop your own code and your own neural network models.

2. Visualize samples from your model. How do they look compared to real data? How is it compared to Figure
2a in the paper?

3. Implement your own GAN with SVHN data. Explore dierent architecture of neural networks and hyperpa-
rameters. Compare samples from your model to real data. How is the quality compared to your GAN on
MNIST? If the training does not go well, what failure modes do you see?

4. (Optional) There are several improved versions of GAN such as Wasserstein GAN (WGAN). Train your own
WGAN https://arxiv.org/abs/1701.07875 on MNIST and SVHN instead of the plain GAN.
