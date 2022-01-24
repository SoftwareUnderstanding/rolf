# Season-Tranfer

## Image to Image Translation

Image to Image to translation is a process whose goal is to learn mappings between input image and output images i.e it is a task which helps to convert an image from domain to another domain. A great amount of research has been done on this subject , most of it has used a supervised approach where we have corresponding pairs of images from the two domains whose mapping we want to learn. The problem with this approach is that for many a tasks there won't be such paired image data. The below image shows image to image translation tasks.

![Here's how image to image translation tasks look like](https://junyanz.github.io/CycleGAN/images/teaser_high_res.jpg)



## Cycle Gan's to the Rescue!

The researcher's at the Berkeley Artificial Intelligence Research (BAIR) published a paper [Unsupervised Image-to-Image translation with cycle consistend adversarial networks](https://arxiv.org/pdf/1703.10593.pdf) in 2017 in which an approach was presented which did not required paired image data for image to image translations tasks. Yes of course still the set of images from both the domains was required they do not need to directly correspond to one another.

## How do they work ? 

### Two Generator's

Cycle Gan is a generative adversarial network with two generators and two discriminators. Let's call one generator G which convert images from domain X to domain Y(**G : X -> Y**)  and other generator F which converts images from domain Y to domain X (**F : Y -> X**). Each generator has a corresponding discriminator which tries to tell apart the synthesized image and real image.

### The Loss Function

There are two loss functions viz Adversarial loss and cycle consistency loss. The adversarial loss would come as no surpirse to people who have worked with GAN's. Both the generators are trying to 'fool' the discriminators i.e they are trying to make them classify fakes images as real ones. The below image shows the adversarial loss

![adversarial loss](https://i.imgur.com/jmPS9NQ.png)

#### The cycle consistency loss (The big gun!)

What does cycle consistent means? For ex : Consider you translate a sentence from English to French and translate it back , you should reach at the original sentence. This is what cycle consistent means. More formally speaking we have a genereator **G:X-->Y**  and another generator **F:Y-->X** the **G** and **F** should be inverse mappings of each other. Using this assumption both the generators **G** and **F** are trained simultaenously with a loss function that encourages **F(G(x)) ≈ x** and **G(F(y)) ≈ y**. The images below shows how cycle consistency looks 

![cycle](https://www.andrewszot.com/static/img/ml/voice_conversion/cyclegan.png)

The cycle consistency loss ensures the property that **x --> G(x) --> F(G(x)) ≈ x**
![cycle loss](https://i.imgur.com/OurehZ5.png)

Combining both the adversarial and cycle consistency loss we get out loss function
![total loss](https://i.imgur.com/PDYywet.png)

The objective of our task is to minmize the combined loss for generators and maximize for the discriminator.
![obj](https://i.imgur.com/QpAYiUy.png)

## The Network Architecture

### Generator Architecture

The Cycle Gan generator is composed of three sections : an encoder , residual blocks , decoder. The input image is fed directly into the encoder which shrinks the size while increasing the number of channels.The encoder consists of three convoltuional layers.The resulting output is the passed to six residual blocks. The output from the residual blocks is expanded by the decoder which comprises of three transpose convolutions.

![Architecture](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/cycle-gan/notebook_images/cyclegan_generator_ex.png)

### Discriminator Architecture

The discriminators are PatchGAN's .PatchGAN's are fully convolutional neural nets that look at a patch of an image and classify it as real or fake as opposed to classifiying the whole image as fake. This approach is more computationally efficient and it allows the discriminator to look more at the surface level features like texture,etc which is more important in image translation tasks.

![Disc Arch](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/cycle-gan/notebook_images/discriminator_layers.png)

-----

## My Results

I trained the net for 4000 epochs with avg time per epoch ≈ 12 secs on GTX 1050Ti.

![results](https://github.com/Atharva-Phatak/Season-Transfer/blob/master/outputs/sample-0004000-X-Y.png)

## References
* Unpaired Image to Image Translation using Cycle Consistent Adversarial Networks:https://arxiv.org/pdf/1703.10593.pdf
* Author's Pytorch Code : https://github.com/junyanz/CycleGAN
* ICCV talk : https://www.youtube.com/watch?v=AxrKVfjSBiA

## Dependencies

* Python 3.6
* PyTorch
* Torchvision
* Numpy
* Scikit-Image









