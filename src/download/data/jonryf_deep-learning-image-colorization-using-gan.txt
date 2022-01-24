# Pix2Face Portrait Colorization

Bringing life to images by applying deep learning to colorize black and white images. This problem is of significant interest for applications including degraded images and aged images. In this project, we will recreate the pix2pix model from the paper "Image-to-Image Translation with Conditional Adversarial Nets" (https://arxiv.org/abs/1611.07004). Pix2pix is a conditional generative adversarial network (cGAN) that learns a dynamic mapping from an input image over to the output image. After 100 epochs training on 2,000 images, the model was able to generate 1,000 colorized test images with an average structural similarity index (SSIM) score of 0.93 when compared to the ground truth images. 

The encoder-decoder consists of a U-Net (https://arxiv.org/abs/1505.04597) neural network with skip connections between the i-thlayer of the encoder and the (n-i)-th layer of the decoder. This helps to pass spatial information fromthe input further along in the output. O


![colorization](colorization.jpg)
