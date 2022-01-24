# Self-Attention GAN in Hearthstone
<p align="center"><img width="100%" src="image/banner.jpg" /></p>
**[Han Zhang, Ian Goodfellow, Dimitris Metaxas and Augustus Odena, "Self-Attention Generative Adversarial Networks." arXiv preprint arXiv:1805.08318 (2018)](https://arxiv.org/abs/1805.08318).**

## Meta overview
This repository provides a PyTorch implementation of [SAGAN](https://arxiv.org/abs/1805.08318). Both wgan-gp and wgan-hinge loss are ready, but note that wgan-gp is somehow not compatible with the spectral normalization. Remove all the spectral normalization at the model for the adoption of wgan-gp. Self-attentions are applied before CNN of both discriminator and generator.

##### Self Attention Layer
<p align="center"><img width="100%" src="image/main_model.PNG" /></p>

## Original Repo status
* Unsupervised setting (use no label yet)
* Applied: [Spectral Normalization](https://arxiv.org/abs/1802.05957), code from [here](https://github.com/christiancosgrove/pytorch-spectral-normalization-gan)
* Implemented: self-attention module, two-timescale update rule (TTUR), wgan-hinge loss, wgan-gp loss

## Current Repo status
- [x] Parallel Computation on multi-GPU
- [x] Tensorboard loggings
- [x] Attention visualization on 64 * 64 image
- [x] Create Attention map of 64 * 64 image (4096 * 4096)
- [x] Change custom ([hearthstone](https://github.com/schmich/hearthstone-card-images)) dataset
- [ ] Create 256*256 image [branch pix256]

##### Warning: 64*64 is the maximum 2power size of attention map for training in 2 Nvidia GTX 1080 Ti (24GB RAM)


## Prerequisites
* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.3.0](http://pytorch.org/)
* [opencv-python](https://pypi.org/project/opencv-python/)
* Details in `requirements.txt`

## Usage

#### 1. Clone the repository
```bash
$ git clone https://github.com/heykeetae/Self-Attention-GAN.git
$ cd Self-Attention-GAN
# for conda user
$ conda create -n sagan python=3.5
$ conda activate sagan
$ conda install pytorch=0.3.0

$ pip install -r requirements.txt
```

#### 2. Install datasets (CelebA or LSUN or Hearthstone)
```bash
$ cd data
$ bash download.sh CelebA (404 not found)
# or
$ bash download.sh LSUN
# For Hearthstone player
$ mkdir hearthstone-card-images
$ cd hearthstone-card-images
$ wget https://www.dropbox.com/s/vvaxb4maoj4ri34/hearthstone_card.zip?dl=0
$ unzip hearthstone_card.zip?dl=0
 ```

#### 3. Train
##### (i) Train in CelebA or Sagan dataset
```bash
$ python main.py --batch_size 64 --imsize 64 --dataset celeb --adv_loss hinge --version sagan_celeb
# or
$ python main.py --batch_size 64 --imsize 64 --dataset lsun --adv_loss hinge --version sagan_lsun
```

##### (ii) Custom parameteric Train in Hearthstone dataset
```bash
$ python main.py --batch_size 16 --imsize 64 --dataset hearthstone --adv_loss hinge --version sagan_hearth_at1 --num_workers 16 --use_tensorboard True --parallel True --total_step 100000 --log_step 100
```
For argument details, please read parameter.py

#### 4. Attention & Statistics visualization
 ```bash
 tensorboard --logdir ./logs/sagan_hearth_at1
 ```

#### 5. Fake images located at
```bash
$ cd samples/sagan_celeb
# or
$ cd samples/sagan_lsun
# or
$ cd samples/sagan_hearth_at1

```
Samples generated every 100 iterations are located. The rate of sampling could be controlled via --sample_step (ex, --sample_step 100).


## 64*64 Results (step #95500)

<p align="center"><img width="100%" src="image/6464_95500.png" /></p>

### 64*64 Attention result on Hearthstone (step #95500)

- Colormap from opencv(https://docs.opencv.org/2.4/modules/contrib/doc/facerec/colormaps.html)
- Most attent part shows in RED (1) , most non-attent part shows in BLUE(0)
- Scores are ranged in [0,1]:
![alt text](https://docs.opencv.org/2.4/_images/colorscale_jet.jpg)

<p align="center"><img width="100%" src="image/6464_95500_attn.png" /></p>

&nbsp;
