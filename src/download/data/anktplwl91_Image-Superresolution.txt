# Image-Superresolution
Different implementations for 8X Image Superresolution of flower images

This repository has my implementations for 8X Super-Resolution of Flower images for a competition held here : https://app.wandb.ai/wandb/superres/benchmark

**Problem Statement**

We are given images for Flowers in 32X32 resolution as input and our objective is to super-resolve these images by 8X to 256X256
resolution.

**Models and Solutions**

I implemented models based on few papers which I read on Image Super-Resolution. Here, I have presented the results which I got
i.e. the super-resolved images as outputs. Following are few papers to which I referred and implemented similar models.

1. Residual Dense Network for Image Super-Resolution (Yulun Zhang et.al.) - https://arxiv.org/pdf/1802.08797v2.pdf
2. A Fully Progressive Approach to Single-Image Super-Resolution (Yifan Wang et.al.) - https://arxiv.org/pdf/1804.02900v2.pdf
3. Real-Time Single Image and Video Super-Resolution Using an EfficientSub-Pixel Convolutional Neural Network (Wenzhe Shi et.al.) - https://arxiv.org/pdf/1609.05158.pdf

I also tried to come up with GAN but was not able to train them properly, still on my To-Do list.

Below are few logs and analysis snapshots from Wandb website for my best submission.

![Training Logs](training_logs.jpeg)

Also. here are some outputs from my best model, in order of : Input 32X32 image - Predicted 256X256 Output - Ground Truth 256X256

![Output Images](out_images.jpeg)
