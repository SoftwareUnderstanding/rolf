
# DCGAN-pytorch
<b>Implementation of Deep Convolutional Generative Adversarial Networks<br>
Based on paper: Unsupervised representation learning with deep convolutional generative adversarial networks</b><br>


# DCGAN Model Architecture
![image](https://user-images.githubusercontent.com/61140071/101329973-69313280-38b5-11eb-876d-e88e3e8a47ad.png)
https://arxiv.org/pdf/1511.06434.pdf
# Install & Usage

### Clone repo
<img src="https://github.com/hectic97/DCGAN-pytorch/raw/main/examples/gitclone.JPG">

### Install requirements
<img src="https://github.com/hectic97/DCGAN-pytorch/raw/main/examples/install_pip.JPG">

### Argparse 
<img src="https://github.com/hectic97/DCGAN-pytorch/raw/main/examples/argparse_help.JPG">

### Default Args 
<img src="https://github.com/hectic97/DCGAN-pytorch/raw/main/examples/default_arg.JPG">

``` python
$ python main.py -lr=0.0002 -dataset=MNIST -batch_size=128 -hidden_dim=128 -z_dim=100 -beta_1=0.5 -beta_2=0.999 -epochs=30 -optimizer=Adam -visual_batch_step=60 -denormalize_img=True
```


# Prerequisites

<sub><sup>
certifi==2020.12.5\
cffi==1.14.4\
cycler==0.10.0\
kiwisolver==1.3.0\
matplotlib==3.3.2\
mkl-fft==1.2.0\
mkl-random==1.1.0\
mkl-service==2.3.0\
numpy==1.19.2\
olefile==0.46\
pandas==1.1.5\
Pillow==8.0.1\
pip==20.3.3\
pycparser==2.20\
pyparsing==2.4.7\
python-dateutil==2.8.1\
pytz==2020.4\
setuptools==51.0.0.post20201207\
sip==4.19.13\
six==1.15.0\
torch==1.6.0\
torchvision==0.7.0\
tornado==6.1\
tqdm==4.54.1\
wheel==0.36.2\
wincertstore==0.2
</sub></sup>


# Development Environments

Python 3.7.6\
pytorch 1.6.0\
NVIDIA RTX 2060\
Ubuntu 20.04.1 LTS

# Results
## Generated image by same noise during training
|MNIST|CelebA|
:-------------------------:|:-------------------------:
![Alt Text](https://github.com/hectic97/DCGAN-pytorch/raw/main/examples/mnist_z_gen.gif)|![Alt Text](https://github.com/hectic97/DCGAN-pytorch/raw/main/examples/celebA_gif.gif)

## Generated image after 30 epoch training
|MNIST|CelebA|
:-------------------------:|:-------------------------:
<img src="https://github.com/hectic97/DCGAN-pytorch/raw/main/examples/gen_image.JPG" width="500" height="500">| <img src="https://github.com/hectic97/DCGAN-pytorch/raw/main/examples/celeba_30epoch.JPG" width="500" height="500">

## Interpolation
|MNIST|
:-------------------------:
|<img src="https://github.com/hectic97/DCGAN-pytorch/raw/main/examples/interpolate.png" width="800" height="800">|
|CelebA|
|<img src="https://github.com/hectic97/DCGAN-pytorch/blob/main/examples/celeba_interpolation_denomalized.png" width="1100" height="800">|



# Reference
[1] Alec Radford & Luke Metz, Soumith Chintala.(2016). 'Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks' arXiv:1511.06434v2<br>
[2] Ian J. Goodfellow et al. (2014).'Generative Adversarial Networks' arXiv:1406.2661v1<br>
[3] Martin Arjovsky et al. (2017). 'Wasserstein GAN' arXiv:1701.07875<br>
