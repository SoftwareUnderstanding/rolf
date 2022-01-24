Sobolev Wasserstein GAN
=====================================

This repo contains a reference implementation for SWGAN as described in the paper:
> Towards Generalized Implementation of Wasserstein Distance in GANs </br>
> [Minkai Xu](https://minkaixu.com/), Zhiming Zhou, Guansong Lu, Jian Tang, Weinan Zhang, Yong Yu </br>
> AAAI Conference on Artificial Intelligence (AAAI), 2021. </br>
> Paper: [https://arxiv.org/abs/2012.03420](https://arxiv.org/abs/2012.03420) </br>

The implementation is built upon the repo [WGAN-GP](https://github.com/igul222/improved_wgan_training), code for reproducing experiments in ["Improved Training of Wasserstein GANs"](https://arxiv.org/abs/1704.00028).

## Prerequisites

- Python, NumPy, TensorFlow, SciPy, Matplotlib
- A recent NVIDIA GPU

## Models

Configuration for all models is specified in a list of constants at the top of
the file. Two models should work "out of the box":

- `python gan_toy.py`: Toy datasets (8 Gaussians, 25 Gaussians, Swiss Roll). 

For the other models, edit the file to specify the path to the dataset in
`DATA_DIR` before running. Each model's dataset is publicly available; the
download URL is in the file.

- `python gan_cifar_resnet.py`: CIFAR-10


## Citing
If you find SWGAN useful in your research, please consider citing the following two papers:

```
@article{xu2020towards,
  title={Towards Generalized Implementation of Wasserstein Distance in GANs},
  author={Xu, Minkai and Zhou, Zhiming and Lu, Guansong and Tang, Jian and Zhang, Weinan and Yu, Yong},
  journal={AAAI Conference on Artificial Intelligence (AAAI), 2021.},
  year={2020}
}
```
```
@article{gulrajani2017improved,
  title={Improved training of wasserstein gans},
  author={Gulrajani, Ishaan and Ahmed, Faruk and Arjovsky, Martin and Dumoulin, Vincent and Courville, Aaron},
  journal={arXiv preprint arXiv:1704.00028},
  year={2017}
}
```
