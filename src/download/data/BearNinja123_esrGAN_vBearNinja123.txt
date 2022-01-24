# esrGAN_vBearNinja123
## Overview <h2>
My implementation of the srGAN (https://arxiv.org/pdf/1609.04802v5.pdf) and esrGAN (https://arxiv.org/pdf/1809.00219.pdf) papers, upscaling 32x32 px images into 128x128 px images. I ran the srGAN model using Keras on Google Colab and esrGAN on Kaggle for about 10 hours each on a dataset of turtle images I collected on Google Images (https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/).
  
  In the picture below, the images in the top, middle, and bottom row are the LR input images, 4x upscaled images, and HR (ground truth) images.
![32x32 LR Image, 128x128 SR Image, 128x128 Ground Truth](/results/32_128.png)

## Repo Files <h2>
This repo contains a trained esrGAN model (one using dense blocks and one using RRDB blocks described in the esrGAN paper), a trained srGAN model, and a trained esrGAN model, with versions for TF 2.3.0 (default version in Google Colab/Kaggle) and TF 2.2.0 (latest release for Anaconda). For fine-tuning your own model, there is an srGAN and esrGAN Jupyter notebook, and you input your low/high-res images in the /train directory.

I also made a script (superResTest.py) to upscale any image 4x if the dimensions of the image is divisible by 32 (e.g 128x128 -> 512x512, 32x64 -> 128x256)
as well as compare the models with each other. I didn't try to hide any artifacts on the edges of the SR images so you'll see faint lines on the upscaled image if you use the script for yourself.\
![128x128->512x512](/results/128_512.png)
## Comparison <h2>
Here's a picture of the models upscaling the same image.
![srGAN vs. esrGAN vs. esrGAN (RRDB)](/results/comparison.png)
