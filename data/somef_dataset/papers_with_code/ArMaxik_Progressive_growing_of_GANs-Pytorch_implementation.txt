# Progressive Growing of GANs for Improved Quality, Stability, and Variation

This is PyTorch implementation of ProgressiveGAN described in paper ["Progressive Growing of GANs for Improved Quality, Stability, and Variation"](https://arxiv.org/abs/1710.10196).

Work is in progress. Equivalent lr layers still don't work well.

# Usage

## Config

Use `config.json` file for set up model before training.

* exp_name - model name
* batch - batch size
* latent - size of latent space vector
* isize - final generating image size
* epochs - number of epochs
* lr_d - lerning rate of discriminator
* lr_g - lerning rate of generator
* lr_decay_epoch - []
* weights - using in `generate.py` script

## Training

To begin trainig use `train.py` script.
* device_ids - GPU ids. Use list for initialize
* device - use GPU or CPU for training
* data_path - path of dataset

Train on cpu:
```sh
python train.py -c config.json -d cpu
```

Train on 2 gpus:
```sh
python train.py -c config.json -d cuda --dev_ids 0 1
```

## Runing

To use generator run `generate.py` script:
```sh
python generate.py -o out/test/ -c config.json -n 20
```

# Example of generating cats

![Training process](https://github.com/ArMaxik/Progressive_growing_of_GANs-Pytorch_implementation/blob/master/illustrations/training.gif?raw=true)

![Generated example](https://github.com/ArMaxik/Progressive_growing_of_GANs-Pytorch_implementation/blob/master/illustrations/result.png?raw=true)

# Compatability

* Python 3.7.3
* PyTorch 1.7.1
* CUDA 10.1
* CUDNN 7.6.3

# Acknowledgement
[1] https://arxiv.org/abs/1710.10196
[2] https://github.com/nashory/pggan-pytorch