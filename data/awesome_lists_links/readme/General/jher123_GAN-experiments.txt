# Animal Art with Wasserstein and Self-Attention GANs
PyTorch implementations of Wasserstein Generative Adversarial Network https://arxiv.org/abs/1701.07875
and a more recent Self-Attention GAN https://arxiv.org/abs/1805.08318 as well webscrapers to download images from flickr and unsplash.

The networks can generate either 64x64 or 128x128 images.
GANs are framework where we want to produce a model distribution that mimics a given target distribution and it consists of a generator which produces a model distribution and a discriminator that distinguishes between the model and target distribution.
In the past few years, GANs have been successful at various image generation tasks, however their training is known to be a tricky process with challenges such as
stability and convergence issues and sensitivity to the choice of hyperparameters. Different types of GANs have been proposed to tackle these issues.
Wasserstein GAN (WGAN) is a popular approach which uses a loss function based on the Wasserstein distance. Such loss metric correlates with the generator’s convergence and image quality, therefore we have a better idea of when to stop training and benefit from improved stability of the optimization process.

The more recent Self-Attention GAN targets another problem which is the difficulty to efficiently model global, long-range dependencies within images. Even though GANs are really good at learning and generating images with complex textures (e.g. landscapes), they struggle with capturing complex geometric or structural patterns that occur consistently in some classes (e.g generating realistic looking animal with a clearly defined head, legs etc). This is due to the convolutional architecture where information is processed in a local neighbourhood and representations learnt in a hierarchical fashion. SAGAN incorporates Self-Attention layers into the Discriminator and Generator architectures in order to help the networks learn long-range dependencies. In addition to that SAGAN authors propose applying spectral normalisation technique to both of the networks' weights as well as two-timescale update rule (different learning rates for the Generator and the Discriminator) to stabilise training.

The goal of this project is to compare visual results achieved from generating images with different types of GANs and experiment with creating artsy looking images of animals using GANs by swapping images for different style images for a few epochs during training.

## Examples
**WGAN**
Lions (64x64)

<p align="center"><img width="50%" src="images/lions_epoch_5000.jpg" /></p>

Swapping lion images for jellyfish images for 10 epochs during training

<p align="center"><img width="50%" src="images/lionsxjellyfish_epoch_4510.jpg" /></p>

**SAGAN**

coming soon

## Prerequisites
The models are meant to be run on CUDA enabled GPU.
The main requirements are Python 3 and packages contained in `requirements.txt`.
For the unsplash scraper, geckodriver and FireFox are also required.

## Getting Started
To download images from Flickr/Unsplash run one of the scraper scripts in `scrapers` providing a keyword and the number of images to download for Flickr:

`python3 flickr_scraper.py --keyword lions --num_images 3000`

`python3 unsplash_scraper.py --keyword lions`

Before training place the image folder in `input_data`.
The training scripts train the network either from scratch or resuming from a checkpoint file.
They saves images every `gen_img_freq` epochs (default 5) and save model and optimiser checkpoints as well as debug info if desired every
`checkpoint_freq` epochs. The full list of arguments is in scripts. Example training a WGAN from scratch:

`python3 train_WGAN.py --bs 128 --im_size 128 --num_epochs 3500 --version_name lions128 --img_folder_name lions --checkpoint_freq 400`

Example training a WGAN from checkpoint saved at the 2000th epoch:

`python3 train_WGAN.py --bs 128 --im_size 128 --num_epochs 3500 --version_name  the lions128 --img_folder_name lions --checkpoint_freq 400 --resume True --resume_epoch_num 2000`

## Acknowledgements
- The flickr scraper is a modified version of https://gist.github.com/yunjey/14e3a069ad2aa3adf72dee93a53117d6
- The idea of swapping images during training to achieve artistic effects inspired by Robbie Barrat's work https://github.com/robbiebarrat/art-DCGAN
