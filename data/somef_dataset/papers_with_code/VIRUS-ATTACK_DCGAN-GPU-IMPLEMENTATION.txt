# DCGAN-GPU-IMPLEMENTATION
DCGAN stands for deep convolutional Generative Adversarial Networks.

The code was implemented based on the following paper  
  >https://arxiv.org/pdf/1511.06434.pdf


Input is a random normal vector “Code” that passes through de-convolution stacks and output an image.
Transposed convolutions are used in DCGAN'S to generate image from the random noise.During training the transposed convoltions weights are learned i.e that the model learned the distribution of the dataset.

The discriminator takes an image as input, passes through convolution stacks and output a probability (sigmoid value) telling whether or not the image is real.

The model was trained for 150 epochs on Tesla M60 GPU and the weights were saved in chechpointWeights.pth file.
No need to train the model, Just use the trained weights for inference.No need to download the dataset.The Q1.ipynb contains code that automatically download the dataset from my dropbox.

This code is part of my coursework for the course "Deep learning" at IIIT sricity.Refer to the problemStatement&Report.pdf for getting clear insights. 
