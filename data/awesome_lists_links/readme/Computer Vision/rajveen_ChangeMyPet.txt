# BigGAN Generators combined with Image Segmentation Model
A Pytorch implementation for replacing a dog/cat image with a GAN(Generative adversarial networks) generated image. The outline(segment mask) of the dog/cat is created using an Image segmentation model.

# Results:
|![alt text](./assets/epoch.png)|
|:--:|
|*1st epoch*|
|![alt text](./assets/epoch19.png)|
|*19th epoch*|

The project is still underway. The results are quite interesting and the model is learning to generate images in the required segment. Regularization and ensembling losses have given more accurate results but we are still exploring other techniques. Our next step is to activate Discriminator so that the images generated look more real. 

# Pretrained Weights 
The pretrained weights are converted from the tensorflow hub modules: 
- https://tfhub.dev/deepmind/biggan-128/2  
- https://tfhub.dev/deepmind/biggan-256/2 
- https://tfhub.dev/deepmind/biggan-512/2  


# References 
paper: https://arxiv.org/abs/1809.11096

https://github.com/ajbrock/BigGAN-PyTorch
