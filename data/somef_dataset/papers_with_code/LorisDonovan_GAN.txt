# GAN
Python implementation of GAN and DCGAN using pytorch\
Based on the following papers:\
GAN - https://arxiv.org/pdf/1406.2661.pdf \
DCGAN - https://arxiv.org/pdf/1511.06434.pdf 

The results from training for 30 epochs is as follows:
![Gif](./Images/anim2.gif)

The original hyperparameters were:
* Epochs= 30
* Input noise size = 100
* Batch size = 256
* Optimizer = Adam
  * lr=2e-4, betas=(0.5, 0.999) for Generator
  * lr=5e-4, betas=(0.5, 0.999) for Discriminator
* Loss function = BCE loss

## About the repo
* ```main.py``` contains the training loop for the GAN
* ```Notes_GAN.md``` contains notes and algorithm from the papers
* remember to check the ```DATA_PATH``` variable, and change it to your local directory
* there is a Generator and Discriminator for both Multilayer perceptron model as well as CNN
* you can train the GAN using either, all you have to do is use ```Generator``` or ```GeneratorConv``` likewise ```Discriminator``` or ```DiscriminatorConv``` as ```generator``` and ```discriminator``` in the training loop
* the training loop will generate the saved models and checkpoints, which you will need for generating gifs and images
* ```viewOutput.py``` contains the code for generating gifs or images of the output of the GAN
* to generate gifs use the ```showAnimation()``` function which takes in the Generator net and device (i.e., cpu or gpu)
* to generate images use the ```showOutputGrid()``` function which takes in the Generator net and device (i.e., cpu or gpu)
