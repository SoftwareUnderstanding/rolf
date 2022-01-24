# ECE471-SRGAN

## Overview
This repository holds the final project deliverable for UVics ECE 471 class. This project is one of many implementations of the paper **Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network** by *Christian Ledig, et al.*

```
@article{DBLP:journals/corr/LedigTHCATTWS16,
author    = {Christian Ledig and
            Lucas Theis and
            Ferenc Huszar and
            Jose Caballero and
            Andrew P. Aitken and
            Alykhan Tejani and
            Johannes Totz and
            Zehan Wang and
            Wenzhe Shi},
title     = {Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial
            Network},
journal   = {CoRR},
volume    = {abs/1609.04802},
year      = {2016},
url       = {http://arxiv.org/abs/1609.04802},
archivePrefix = {arXiv},
eprint    = {1609.04802},
timestamp = {Mon, 13 Aug 2018 16:48:38 +0200},
biburl    = {https://dblp.org/rec/bib/journals/corr/LedigTHCATTWS16},
bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

Our approach was to imitate the implementation of this paper, and to see its significance to real-world noise. We created our own dataset for testing our theory that the results of the GAN was undoing the downsampling, ie. undoing the gaussian noise.

Unfortunately due to the time constraint from other classes and the magnitude of this project, the model has a flaw which makes the network not converge towards meaningful results. This is possibly due to normalization problems, or a problem with the custom convolution as PyTorch does not have an option for 'same' padding. It is also possible that we are misinterpreting the Content Loss (VGG19 network).

Overall this was a great learning experience and I personally hope to recreate/modify this in the future.

## Get Started

The following information is needed in order to get this project up and running on your system.

### Environment

1. Create a `virtualenv` using `virtualenv python=python3 .venv`. Run `source .venv/bin/activate` to start the environment, and `deactivate` to close it.
2. Install dependencies using `pip install -r requirements.txt`

### Data

The high resolution images that were used during training were from: http://image-net.org/challenges/LSVRC/2017/download-images-1p39.php

>@article{ILSVRC15,
Author = {Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei},
Title = {{ImageNet Large Scale Visual Recognition Challenge}},
Year = {2015},
journal   = {International Journal of Computer Vision (IJCV)},
doi = {10.1007/s11263-015-0816-y},
volume={115},
number={3},
pages={211-252}
}

### Command Line Arguments

There are a few useful command line arguments.

- `data_dir`: (String) Path to image data
- `cropsize`: (Int) cropped size of HR image
- `pretrain`: (Bool) pretrain the generator prior to SRGAN
- `pretrain_epochs`: (Int) Number of pretraining epochs

A full list of command line parameters can be found in `config.py`

### Train

This project does not have the pretrained SRResNet (MSE pixelwise loss function), so pretraining is needed if this is the first run

**Note:** Once pretraining is done, the program will automatically begin training SRGAN

1. Run `python3 main.py pretrain True` to being the program
2. Grab a coffee or tea because this part takes some time

If the model has been pretrained before, then only `python3 main.py` is needed

### Tensorboard

This project uses tensorboard for viewing the training loss and images. In a separate terminal window, run `tensorboard --logdir logs` to run tensorboard locally. With this, you can view the loss in the *SCALARS* tab, and images in the *IMAGES* tab.