
# Welcome to GANSCAPE STUDIO.

### PROBLEM STATEMENT
Online sales of antique and unique art were reported at $6 billion and growing by 11% each year. For Upper middle class families to medium scale business, it can be very tricky to buy exclusive and unique art.

### SOLUTION
To offer unique and exclusive art at a very modest price, I have developed a web application which offers machine generated abstract landscape art which you can buy for $2. The art gallery refreshes every 15 minutes, the art can be yours or lose it forever.

### PRESENTATION LINK
https://drive.google.com/drive/u/1/folders/1az6J5ucolVTDz3x3ktcrNeIYuoErPwZi

## Requirement:  Flask, 1 GPU, Pytorch, Python 3.6, Matplot, NumPy


## Results

Quick view of progressive Training. 
![Insight-ExclusiveArtZoo](Readme%20Images/ezgif.com-gif-maker.gif)

### 150 epoch- 5th Depth - Batch size 32 
![Insight-ExclusiveArtZoo](Readme%20Images/gen_5_150_100.png)



## Dataset Details:

There are two classes landscape and abstract on which the GAN model is trained. 
Approximately GAN model requires 3000 images to completely learn one class.  

![Insight-ExclusiveArtZoo](Dataset.JPG)

## Progressive GAN:
The progressive growing of GANs trains the GAN network in multiple phases. In phase 1, it takes in a latent feature z and uses two convolution layers to generate 4×4 images. Then, we train the discriminator with the generated images and the 4×4 real images. Once the training stables, we add 2 more convolution layers to upsampling the image to 8×8 and 2 more convolution layers to downsampling images in the discriminator. 

I have 6 phases in total, to generate the 128 x 128 abstract landscape images

![ProGAN](ProgressiveGAN.png)

## Use Pretrained weights
GAN_GEN_SHADOW_5.pth and GAN_GEN_5.pth are two pre trained generator network which you can directly use to generate various types of images, including 128x128 landscape oil paintings. 

Import pro_gan generator into your code as shown in GanScapeStudio.py and load .pth trained weights into the generator and generate abstract landscape images. 


### How to Use the project for your custom dataset:
To train on your custom dataset using Progressive Growing of GANs, there is an example in Notebook/trainpgGAN.ipynb
Set hyperparameteres like depth of training model, number of epochs, fade ins, batch size and feed back factor.

#### Refrences
ProGAN- https://arxiv.org/abs/1710.10196 <br/>
Animesh Karewar Blog- https://medium.com/@animeshsk3/the-unprecedented-effectiveness-of-progressive-growing-of-gans-37475c88afa3 <br/>
DCGAN PyTorch- https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html <br/>
Wiki- https://www.wikiart.org/en/paintings-by-genre/landscape?select=featured#!#filterName:featured,viewType:masonry <br/>

This repository is only for educational purpose created during Insight Data Science fellowship.

## Thankyou
