# Deep Image Prior with Transfer Learning

## Intro
Project conducted with: Anirudh Singh Shekhawat, Aishma Raghu, Michelle Sit for CSE 253 (Neural Networks for Pattern Recognition) with Prof. Gary Cottrell.

This project consists in:
1) An implementation of Ulyanov et al's "Deep Image Prior" work (https://dmitryulyanov.github.io/deep_image_prior).
2) An extension to transfer learning across images in inpainting

## Files
  - The notebook
  - A pdf of the notebook (since the notebook doesn't load in the browser)
  - The pdf report in NIPS format

## Deep Image Prior
Deep image prior is a convolutional neural network with a fixed input. The structure of the network functions as a prior for natural images; the network only trains on one image to complete an image enhancement task.

We have implemented two of those tasks based on the original paper:
- In superresolution, the goal is to enhance an image by increasing its resolution. The output of the network has width and height equal to 4x the original image's width and size. The network's output image is downsampled (in a differentiable way) and the loss is computed against the original image itself.
- In inpainting, the goal is to "fill in" a region that is corrupted. In our implementation, we assume the filter defining the corrupt pixels is given. The network's output is the same size as the original image, and the loss is computed only for non-corrupt pixels. The network "hallucinates" the corrupt pixels and achieves impressive results.

## Transfer Learning
Can this set-up be used more efficiently? For example, in video data, a lot of the images are very similar to each other, and if we wanted to remove subtitles from a movie, we shouldn't have to rerun the network with random weight initialization every time.

Here, we tested and confirmed the hypothesis that using the weights from the previous image train faster than randomly initialized weights. Averaging over weights of successive images also yields faster learning; we experiment with a few schemes.

Two possible avenues for future work:
- Using the weights of a network to encode in image can be used for image compression. In typical ML compression schemes, an autoencoder is used, with the encoder creating the compressed representation and the decoder "unpacking" it. Here the training would be the process by which the compressed representation is created, and the compressed representation would be the weights of the network.
- Using DARTS (Differentiable Architecture Search: https://arxiv.org/abs/1806.09055) to learn a network that trains faster for new images in a long sequence of images

## Other
Subtitle Pixels Extraction: A portion of the code is dedicated to creating a filter that selects the pixels that are in the subtitle.

## Using this code

This code was written in Google Colaboratory so it will be easiest to first run there. Some of the image files loaded from Google Drive  may not be publicly accessible; please email samrsabri@gmail.com if you encounter any issues.
