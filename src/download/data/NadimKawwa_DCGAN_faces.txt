# Face Generation With Deep Convolutional Generative Adversarial Networks
![faces_intro](https://github.com/NadimKawwa/DCGAN_faces/blob/master/plots/cover_dcgan.png)

In this repository we will attempt to generate realistic faces from the [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) using deep convolutional generative adversarial networks (DCGANs) per Radford et al [1]. The code is implemtend in python 3.x


## Dataset

CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset with more than 200K celebrity images, each with 40 attribute annotations. The images in this dataset cover large pose variations and background clutter. CelebA has large diversities, large quantities, and rich annotations, including

- 10,177 number of identities,
- 202,599 number of face images, and
- 5 landmark locations, 40 binary attributes annotations per image.

The dataset can be employed as the training and test sets for the following computer vision tasks: face attribute recognition, face detection, landmark (or facial part) localization, and face editing & synthesis.

![data_overview](https://github.com/NadimKawwa/DCGAN_faces/blob/master/plots/overview.png)

## Libraries

- pickle
- numpy
- matplotlib
- torch
- torchvision

It is recommended to have GPU access to optimize training time.


## Methodology

The first step is batching the neural network data using DataLoader class from torchvision. A function is defined to load the images from the root directory and convert them to torch tensors, the user can determine the size of the images.

We then define the discriminator: this is a convolutional classifier, only without any maxpooling layers. To deal with this complex data, it's suggested to use a deep network with normalization such that:
- The inputs to the discriminator are 32x32x3 tensor images
- The output should be a single value that will indicate whether a given image is real or fake

![sketch_discriminator](https://github.com/NadimKawwa/DCGAN_faces/blob/master/plots/conv_discriminator.png)

The generator should upsample an input and generate a new image of the same size as our training data 32x32x3. This should be mostly transpose convolutional layers with normalization applied to the outputs.
- The inputs to the generator are vectors of some length z_size
- The output should be a image of shape 32x32x3

![sketch_generator](https://github.com/NadimKawwa/DCGAN_faces/blob/master/plots/conv_generator.png)

To help the models converge, we initialize the weights of the convolutional and linear layers in the model. We define a function such that:
- Initialize only convolutional and linear layers
- Initialize the weights to a normal distribution, centered around 0, with a standard deviation of 0.02.
- The bias terms, if they exist, may be left alone or set to 0.


We also need to calculate the losses for both types of adversarial networks. For the discriminator, the total loss is the sum of the losses for real and fake images:
d_loss = d_real_loss + d_fake_loss.

We want the discriminator to output 1 for real images and 0 for fake images, so we need to set up the losses to reflect that.

The generator loss will look similar only with flipped labels. The generator's goal is to get the discriminator to think its generated images are real.

Training will involve alternating between training the discriminator and the generator. We will use the  functions real_loss and fake_loss to help  calculate the discriminator losses.
Wew will train the discriminator by alternating on real and fake images
Then the generator, which tries to trick the discriminator and should have an opposing loss function

The learning rate used here is lr=0.0002


## Results

Our initial data was a dataset of real celebrity imges:
![real_images](https://github.com/NadimKawwa/DCGAN_faces/blob/master/plots/processed_face_data.png)


The plot below tracks generator and discriminator loss through the training epochs.
![training_loss](https://github.com/NadimKawwa/DCGAN_faces/blob/master/plots/training_loss.png)

We notice that the output resembles human like features, yet is not close to the real thing.

![fake_images](https://github.com/NadimKawwa/DCGAN_faces/blob/master/plots/fake_faces.png)

The model can definitely improved upon with more experimentation and tips from Soumith Chintala[2].


## References
[1] https://arxiv.org/abs/1511.06434
[2] https://github.com/soumith/ganhacks

Credit to Udacity for the notebook layout, cover photo, unit tests, and sketches of Discriminator and Generator networks.
