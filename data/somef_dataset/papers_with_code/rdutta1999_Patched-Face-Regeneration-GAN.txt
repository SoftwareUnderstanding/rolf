# Patched-Face-Regeneration-GAN

## Introduction
In this project, my aim was to develop a model that could regenerate patched/covered parts of human faces, and achieve believable results. I used the [Celeb-A](https://www.kaggle.com/jessicali9530/celeba-dataset) dataset, and created a Generative Adversarial Network with a Denoising Autoencoder as the Generator and a Deep Convolutional Network as the Discriminator. I chose this architecture based on *Avery Allen and Wenchen Li*'s [Generative Adversarial Denoising Autoencoder for Face Completion](https://www.cc.gatech.edu/~hays/7476/projects/Avery_Wenchen/).

The Denoising Autoencoder has 'relu' activations in the middle layers while the output layer had a 'tanh' activation. Each Convolution layer was followed by a BatchNormalization layer. The Discriminator has 'LeakyReLU' activations for the Convolution part, with a BatchNormalization layer following every Conv layer. At the end, the output from the CNN segment was flattened and connected to a Dense layer with 1 node, having 'sigmoid' as the activation function. This would enable the discrimator to predict the probability that the input image is real.

I added distortions to the images in two ways:-
- Added random Gaussian noise.
- Added random sized Black Patches.

The entire training was done on a GTX 1080 GPU, and took about 12days.

The latest checkpoints and the saved generator and discriminator can be found [here](https://drive.google.com/drive/folders/13wUgCcENajkPZ4MHz2bHrJtQepyVDvtb?usp=sharing).

A few sample generated images are present in `saved_imgs`.

## Setting Up
1) Create a new Python/Anaconda environment (optional but recommended). You might use the `environment.yml` file for this purpose (Skip Step-2 in that case).

2) Install the necessary packages. Refer to the packages mentioned in `environment.yml`.

3) Download the training checkpoints and saved generator and discriminator models from [here](https://drive.google.com/drive/folders/13wUgCcENajkPZ4MHz2bHrJtQepyVDvtb?usp=sharing).

4) Download the [Celeb-A](https://www.kaggle.com/jessicali9530/celeba-dataset) dataset, and place it in the directory in the following manner:-
<pre>
├─── Patched-Face-Regeneration-GAN
     ├─── ..
     ├─── saved_imgs     
     ├─── training_checkpoints
     ├─── saved_discriminator
     ├─── saved_generator
     ├─── discriminator.png
     ├─── generator.png
     ├─── inference_output.png
     ├─── environment.yml
     ├─── Face_Generation.ipynb    
     └─── images 
           └─── img_align_celeba
               ├─── 000001.jpg
               ├─── 000002.jpg
               ├─── 000003.jpg
               ├─── 000004.jpg
               ├─── 000005.jpg
               ├─── ..
               ├─── ..
               ├─── 202597.jpg
               ├─── 202598.jpg
               └─── 202599.jpg
</pre>

## Output
In the following figure, the first row shows the Actual images, the second row shows the Patched/Distorted Images and the last row shows the images produced by the generator.
![Pic1](inference_output.png?raw=true)

Some more images produced by the generator:- 

![Pic1](./saved_imgs/image997_1.jpg?raw=true) ![Pic2](./saved_imgs/image997_2.jpg?raw=true) ![Pic3](./saved_imgs/image997_3.jpg?raw=true) ![Pic4](./saved_imgs/image997_4.jpg?raw=true) ![Pic5](./saved_imgs/image997_5.jpg?raw=true) ![Pic6](./saved_imgs/image998_1.jpg?raw=true) ![Pic7](./saved_imgs/image998_2.jpg?raw=true) ![Pic8](./saved_imgs/image998_3.jpg?raw=true) ![Pic9](./saved_imgs/image998_4.jpg?raw=true)  ![Pic10](./saved_imgs/image998_5.jpg?raw=true) ![Pic11](./saved_imgs/image999_1.jpg?raw=true) ![Pic12](./saved_imgs/image999_2.jpg?raw=true) ![Pic13](./saved_imgs/image999_4.jpg?raw=true) ![Pic14](./saved_imgs/image999_5.jpg?raw=true) ![Pic15](./saved_imgs/image1000_1.jpg?raw=true) ![Pic16](./saved_imgs/image1000_2.jpg?raw=true) ![Pic17](./saved_imgs/image1000_3.jpg?raw=true) ![Pic18](./saved_imgs/image1000_4.jpg?raw=true)

## References
- https://www.cc.gatech.edu/~hays/7476/projects/Avery_Wenchen/
- https://www.tensorflow.org/tutorials/generative/dcgan
- https://arxiv.org/abs/1406.2661
