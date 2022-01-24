# Generative-Adversarial-Nets
Different GAN ([Generative Adversarial Network](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)) architectures in TensorFlow

## w gan (*./w_gan/*)

### Trained on Digimon images:
![](images/digi_gan_1.jpg)

![](images/digi_gan_2.jpg)

### Trained on the original 150 Pokemon:
![](images/poke_example.jpg)

### Wasserstein GAN
https://arxiv.org/abs/1701.07875

### Generator
tf.layers.conv2d_transpose
tf.contrib.layers.batch_norm
### Discriminator
tf.layers.conv2d
tf.contrib.layers.batch_norm

def leaky_relu(input, name, leak=0.2):
    return tf.maximum(input, leak * input, name=name)

w- gan











## GAN (*/vaniall_gan/*)
A Generative Adversarial Net implemented with **TensorFlow** using the
**MNIST** data set.

#### Generator:
* Input: **100**
* Output: **784**
* Purpose: Will learn to **output images** that **look** like a **real**
image from **random input**.



#### Discriminator:
* Input: **784**
* Output: **1**
* Purpose: Will learn to tell a **real** ("looks like it could be a real image in MNIST dataset") **image**(784) from a fake one.


#### Notes and Outputs
A problem with the way that I built this is that I used the **same architecture**
for **both** the **generator** and **discriminator**. Although I thought this save me, the developer, a lot of time it actually
caused a lot of problems with trying to pigeonhole that architecture to work with a smaller input **(Discriminator: 28x28 vs 10x10 : Generator)**.

##### Architecture

* conv1 -> relu -> pool ->
* conv2 -> relu -> pool ->
* conv3 -> relu -> pool ->
* fullyConnected1 -> relu ->
* fullyConnected2 -> relu ->
* fullyConnected3 ->

100 random numbers -> Generator -> ImageOutput -> Discriminator -> (Real|Fake)


![generated gan output](images/gan_generated.gif)

![](images/init.png)
![](images/two.png)
![](images/two1.png)
![](images/three.png)
![](images/weird.png)
![](images/weird2.png)
![](images/weird3.png)

## ColorGan