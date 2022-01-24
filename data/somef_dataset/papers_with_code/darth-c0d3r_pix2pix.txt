# pix2pix
CS 663 (Digital Image Processing) Project

## Team
#### Rajat Rathi (160050015)
#### Anmol Singh (160050107)
#### Gurparkash Singh (160050112)

## Overview

The problem we are trying to solve is a general conversion of black and white images to colored. For this, we will be using **Conditional GANs** [Generative Adversarial Networks] to implement and extend the **pix2pix** network, which is a general neural network model to learn a mapping from one set of images to another, which, in our case will be from the set of black and white images to colored images.

Our project can be easily extended to other tasks such as day to night, labels to street scene, deblurring images etc. as the basic pix2pix network is shared among all such applications. If time permits, we will try to apply our network to one of the other tasks as well. We will be using the PyTorch framework in Python to implement the network.

## Research Papers Referred
### Image-to-Image Translation with Conditional Adversarial Networks [1]

This is the Research Paper written on the pix2pix network written at the Berkeley AI Research (BAIR) Laboratory, UC Berkeley. This is the research paper that we are trying to replicate and apply to the task of colorizing Black and White images.

### Generative Adversarial Nets [2]

This is the first paper on Generative Adversarial Networks by Ian Goodfellow et al.

## Datasets

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.

### MIT CVCL Dataset [3]

Contains multiple databases composed of scenes belonging to the same semantic category. All images are of size 256x256, in jpeg format.

## Evaluation Metrics

The evaluation metric that we have used to compare the results is SSIM (Structural Similarity Index) [4] It compares two images and gives an output between 0 and 1. 1 meaning that images are exactly same and 0 meaning that they are completely different. We use it to compare the generated images and the ground truth images.

## Results

For results, please refer the final presentation in the report folder.

## How to run

### Creating a dataset

First create a folder `datasets` in the main repository. Create a folder inside that with the name of dataset you are using. Inside that folder, create a folder `full_dataset` that contains all the colored images. After this run the following command.
```bash
python3 create_dataset.py <Dataset Name> <Task>
```
The task can be one of `bnw2color, impaint_fix, deblur`.

### Training the model

Before training, create a folder named `saved_models` in which the results will be automatically saved. After successfully creating the dataset and creating the folder, run the following command to start training the model.
```bash
python3 train.py <Dataset Name> <Task>
```
The hyperparameters can be tuned by making the relevant changes in `train.py`, `generator.py` and `discriminator.py` files.

### Using the trained Model

To use the trained model on eval or test set, use the following command.
```bash
python3 predict.py <Dataset Name> <Task>
```
To switch between eval and test sets make relevant changes in the path variable in the `predict.py` file.

### Evaluation Metrics

Use the `eval_metrics.py` file to get the evaluation results on the images in output folder.

## References:

<ol>
	<li>pix2pix Research Paper: https://arxiv.org/pdf/1611.07004.pdf</li>
	<li>GAN Research Paper: https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf</li>
	<li>MIT CVCL Dataset: http://cvcl.mit.edu/database.htm</li>
	<li>SSIM: https://en.wikipedia.org/wiki/Structural_similarity</li>
	<li>Official pix2pix Repo: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix</li>
	<li>PyTorch Website: https://pytorch.org/</li>
</ol>

## Tutorials:
<ul>
	<li>Intro to GANs: https://medium.freecodecamp.org/an-intuitive-introduction-to-generative-adversarial-networks-gans-7a2264a81394</li>
	<li>Generative Models (Stanford): https://www.youtube.com/watch?v=5WoItGTWV54&index=13&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv</li>
	<li>GANs in PyTorch (Simple): https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f</li>
	<li>GANs in PyTorch (Official): https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html</li>
	<li>pix2pix Tutorial: https://towardsdatascience.com/cyclegans-and-pix2pix-5e6a5f0159c4</li>
	<li>Colorization with GAN Repo: https://github.com/ImagingLab/Colorizing-with-GANs</li>
	<li>GAN Hacks Repo: https://github.com/soumith/ganhacks</li>
	<li>Installing CUDA 9.0: https://gist.github.com/zhanwenchen/e520767a409325d9961072f666815bb8</li>
	<li>PyTorch Examples: https://cs230-stanford.github.io/pytorch-getting-started.html</li>
</ul>
