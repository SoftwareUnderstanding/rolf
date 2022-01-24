# BERGAN: music bar generation and techno music with GANs

work in progress, project submitted to an open call

minimal requirements, can be installed with pip in a python3 virtualenv (pip install -r requirements.txt)

these requirements do not cover the data preparation pipelines described at the bottom of the readme

**_ current application is generating techno music from latent interpolations**

**--> each latent point maps to a music bar**

**_ next application is "inpainting" e.g. generation from loudness or transient curve to audio**

**--> convert acoustic drum loop or hand tapping to techno music bars**


## GANs ONLY EXPERIMENT

code base = __export_interp.py + __nn_utils.py + __train.py

experiment to make techno music with GANs, default is to train on 2 sec. audio clips at 16kHz = 1 bar 4/4 at 120BPM

unconditional generator and multi-scale discriminators, default working config = WGANGP_2scales_WN_WN_crop0.gin

without GP the GANs are very prone to mode collapse/overfitting, also avoid BN discriminator with GP

the corresponding configs have suffixes = WGAN or LSGAN = wasserstein or least-square


## AEs+GANs EXPERIMENT

code base = __export_interp_ae.py + __nn_utils_ae.py + __train_ae.py

train as a AE/GAN (or VAE/GAN or WAE/GAN) to avoid mode collapse of GAN = more stable training without GP and possibly better mode coverage

use deep feature reconstruction in discriminator activations and optional multi-scale spectrogram reconstruction

VAE/GAN adds the KLD regularization to the encoder and WAE/GAN adds the MMD regularization = both impose a gaussian prior for sampling and interpolation

the corresponding configs have suffixes = AE_GAN or VAE_GAN (variational ae) or WAE_GAN (wasserstein ae)

the GAN is only trained as least-square without gradient penalty


## TODO

try training at 32kHz

try other music genres with 4/4 musical structure

make a google colab demo (with pretrained models to run in __export_interp.py)


## AUDIO SAMPLES (GANs only)

examples of random linear interpolations with 20 points equally spaced in the generator latent space = 20 bars = 40 sec.

training data is between 5.000 and 20.000 examples of bars extracted from recordings of the "Raster Norton" label

https://raster-media.net (I do not own copyrights, this is an independent research experiment)

models were trained for 48 hours on a single V100 GPU (a 12GB GPU is fine too) ; sampling of 40 sec. on Macbook Pro CPU (2015) takes about 3 sec. so the inference speed is reasonable

raw audio outputs of the models at 16kHz --> https://soundcloud.com/adrien-bitton/interpolations


## GAN TRAINING

optimize the generator to sample realistic 1 bar audio of 2 sec. (120BPM) at SR=16kHz (extendable to 32kHz or 48kHz)

<p align="center">
  <img src="./figures/bergan_gan_train.jpg" width="750" title="GAN training">
</p>


## AUDIO SAMPLES (AEs+GANs)

to come


## AEs+GANs TRAINING

figure to come


## GENERATION

sample series of 1 bar audio along a random linear interpolation and concatenate the generator outputs into a track at fixed BPM with progressive variation of rhythmic and acoustic contents

<p align="center">
  <img src="./figures/bergan_interp.jpg" width="750" title="generator interpolation">
</p>


## RELATED PAPERS AND REPOSITORIES

MelGAN

https://arxiv.org/abs/1910.06711

https://github.com/seungwonpark/melgan

WaveGAN

https://arxiv.org/abs/1802.04208

https://github.com/mostafaelaraby/wavegan-pytorch

nice review of GAN frameworks

https://arxiv.org/abs/1807.04720

the AEs+GANs framework 

https://arxiv.org/abs/1512.09300


## ACKNOWLEDGEMENTS

thanks to Philippe Esling (https://github.com/acids-ircam) and Thomas Haferlach (https://github.com/voodoohop) for their help in developping the data preparation pipelines

data preparation aims at extracting music bars aligned on the downbeat and stretching them to the target BPM

we either rely on python packages (e.g. librosa, madmom) or on parsing warp markers from Ableton .asd files (https://github.com/voodoohop/extract-warpmarkers)

thanks as well to Antoine Caillon (https://github.com/caillonantoine) for insightful discussion on the challenges of training GANs

and thanks to IRCAM and Compute Canada for the allowed computation ressources for training models
