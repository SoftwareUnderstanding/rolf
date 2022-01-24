# DCGAN


Deep Convolutional GAN are nothing but GAN's with Deep Convolutional layers. Google has released a pre-built model for MNIST digit generation using DCGAN, you can find out more from the link below. Here, I am trying to take it one step forward and create a DCGAN model for CelebA dataset using Tensorflow 2.0. Note that the this time number of dimensions and channels would be different as oppose to MNIST because CelebA dataset has coloured images. For this, I had to create the Generator again which can generate coloured Images. 
I could not train the model for long time so I would highly appreciate if anyone can run the model and train it for more epochs if you have GPU.


Paper: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks : https://arxiv.org/abs/1511.06434

Link for DCGAN implementation using MNIST dataset in Tensorflow 2.0 : https://www.tensorflow.org/beta/tutorials/generative/dcgan

Link to the YouTube Video to make things clear: https://www.youtube.com/watch?v=jIP6YZUsj-o&t=335s

# CGAN

Conditional GAN's are advanced form of GAN's where we provide conditions as well with the random noise to generate the data. I am using MNIST dataset and trying to tune the GAN so that it will generate the data based on conditions such as Images of numbers from 0 to 9, for that we will have to add random noise (as always) + conditions such as 0-9, this 0-9 numberic value is then converted and added into the noise which can be used by the Generator. Desciminator will then decide if the image looks like it is actually drawn from the sample or not. 

Reference: https://github.com/miranthajayatilake/CGAN-Keras

Paper : Conditional Generative Adversarial Nets :  https://arxiv.org/abs/1411.1784

Do let me know if you have any questions regarding any of the above code, I would be happy to help you.

Isaac,

https://www.linkedin.com/in/isaac-patole/
