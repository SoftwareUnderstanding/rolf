# gans
This repoitory contains projects undertaken in an attempt to understand gans, and implement them in many different areas.
The projects start from very simple sequence generation, and I won't like to place a limit on the other end. SKY IS THE LIMIT 
(or probably something beyond sky in our age :))

## Going Through Projects and Possibly Implementing Projects Yourselves
I plan to implement most of these projects in tensorflow. In most cases, installing tensorflow should be sufficient.

### Sequence Generation
This is a very simple implementation of the GAN paper by Ian Goodfellow (Generative Adversarial Nets) https://arxiv.org/abs/1406.2661
We use a very simple gan network to mimic a random mathematical distribution created using numpy.

### fashion-mnist
This project generates images which mimic the distribution of images in the popular fashion-mnist dataset

### conditional-gan
Whereas the previous project generated images from the sample space of fashion-mnist dataset, we had no control over what image we wanted to generate. This GAN accepts a parameter, which can be used to generate a specific type of image, say (image of a sandal)

![One of the images while training](https://github.com/vaibhav369/gans/blob/master/conditional_gan/75.jpg)

### 3d shape-generation
In the rest of the projects, we work with the familiar pixels. Here, the only major difference is that we are working with voxels i.e. volumetric pixels. These voxels are used to construct 3d images. Using techniques similar to fashion-mnist, we generate 3d images
