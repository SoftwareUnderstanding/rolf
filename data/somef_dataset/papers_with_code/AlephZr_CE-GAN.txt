# CE-GAN
## Prerequisites

- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Preparation
- Preparing *.npz* files for Pytorch Inception metrics evaluation (cifar10 as an example):
```
python envs_repo/inception_pytorch/calculate_inception_moments.py --dataset C10 --data_root datasets
```
- Preparing *.tgz* files for Tensorflow Inception metrics evaluation
- Preparing *.npz* files for Tensorflow FID metrics evaluation

### Operator GANs Training

An example of [GAN](https://arxiv.org/abs/1406.2661) training command was saved in [./scripts/operator_cifar10.sh](). Train a model (cifar10 as an example): 
```bash
bash ./scripts/operator_cifar10.sh
```

### CE-GAN Training

An example of CE-GAN training command was saved in [./scripts/cegan_cifar10.sh](). Train a model (cifar10 as an example):
```bash
bash ./scripts/cegan_cifar10.sh
```

## Acknowledgments

Pytorch Inception metrics code from [BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch).

TensorFlow Inception Score code from [OpenAI's Improved-GAN.](https://github.com/openai/improved-gan).

TensorFlow FID code from [TTUR](https://github.com/bioinf-jku/TTUR).

