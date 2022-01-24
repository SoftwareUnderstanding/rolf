# Hyperspectral ImageReconstruction with GAN

Based on the Image Super-Resolution GAN https://arxiv.org/pdf/1609.04802.pdf We implemented a 
GAN to create an end-to-end optimization framework for hyperspectral image reconstruction with a custom layer that creates a convolution between the hyperspectral 
image and a group of PSFs.

1. GAN_Reconstruction --> is a replique of the original paper thar you can find in this repository https://github.com/deepak112/Keras-SRGAN, with 3 channels 
2. GAN_UIS --> is our propose with the custom layer and 12 channels 
3. Titan_GAN_UIS --> is a colab with the hyperspectral image reconstruction  with 31 chanels 
4. GAN_FRAMEWORK --> is the final framework with 70 channels

