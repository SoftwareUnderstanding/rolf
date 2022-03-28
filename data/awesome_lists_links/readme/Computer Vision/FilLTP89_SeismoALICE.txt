# SeismoALICE

Adversarially Learned Inference (ALI) + Conditional Entropy (CE) for prediction of seismic signals based on Physics-Based numerical simulations.

> [Gatti, F.](https://github.com/FilLTP89) and Clouteau, D. (2020). "Towards blending Physics-Based numerical simulations and seismic databases using Generative Adversarial Network". Submitted at Computer Methods in Applied Mechanics and Engineering, Special Issue: "AI in Computational Mechanics and Engineering Sciences". 
This work was inspired by the following original works:

[Adversarially Learned Inference with Conditional Entropy (ALICE)](https://github.com/ChunyuanLI/ALICE)
> Li, C. et al. (2017). "ALICE: Towards Understanding Adversarial Learning for Joint Distribution Matching". Duke University. NIPS, 2017. [https://arxiv.org/abs/1709.01215](https://arxiv.org/abs/1709.01215)

[DCGAN.torch: Train your own image generator](https://github.com/soumith/dcgan.torch)
> Radford, A. et al. (2015). "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks". ICLR 2016. [https://arxiv.org/abs/1511.06434](https://arxiv.org/abs/1511.06434)

[ANN2BB: NN-based broadband ground motion generator from 3D earthquake simulations](https://github.com/FilLTP89/ANN2BB.git)
> Paolucci, R., Gatti, F. et al. (2018). "Broadband Ground Motions from 3D Physics-Based Numerical Simulations Using Artificial Neural Networks". BSSA, 2018 [htt    ps://doi.org/10.1785/0120170293](https://doi.org/10.1785/0120170293)

## Prerequisites

- Computer with Linux or OSX
- Torch

For training, an NVIDIA GPU is strongly recommended for speed. CPU is supported but training is very slow. GPU-based version was tested on `cuda 9.0`, `cuda 9.2`.

### Installing dependencies

Before getting started, some prerequisites must be installed via `pip`:

```
pip install -r requirements.txt
```

Before running the scripts, it is necessary to add `./src` to the `PYTHONPATH`:

```
export PYTHONPATH="./src"
```

## Getting Started

Three Deep-Convolutional Adversarial AutoEncoders (DCAAE) can be trained and tested, according to the reconstruction frequency band:
    
 1. `broadband` (`bb`) seismic signals (0-30 Hz)
 2. `filtered` (`fl`) seismic signals (0-<img src="/tex/6351b053cd3fdab4b400a9c29ea5f732.svg?invert_in_darkmode&sanitize=true" align=middle width=18.28248014999999pt height=22.831056599999986pt/> Hz, where fc can be set via `--cutoff` option
 3. `hybrid` (`hb`) seismic signals (0-<img src="/tex/6351b053cd3fdab4b400a9c29ea5f732.svg?invert_in_darkmode&sanitize=true" align=middle width=18.28248014999999pt height=22.831056599999986pt/> + <img src="/tex/6351b053cd3fdab4b400a9c29ea5f732.svg?invert_in_darkmode&sanitize=true" align=middle width=18.28248014999999pt height=22.831056599999986pt/>-30 Hz)

The original contribution of this paper is `hybrid` `DCAAE`, which takes a low-frequency signal (0-<img src="/tex/6351b053cd3fdab4b400a9c29ea5f732.svg?invert_in_darkmode&sanitize=true" align=middle width=18.28248014999999pt height=22.831056599999986pt/> Hz) and reconstructs the broad-band one.

Each `DCAAE` can undergo three different ``actions`` (to be listed in the `actions.txt` file [True/False])

 1. `tract`: train 
 2. `trplt`: plot/generate
 3. `trcmp`: compare with ANN2BB

Each action implies the choice of a corresponding `strategy` (to be specified in the `strategy.txt` file) for keywords `encoder`,`decoder` (...and others, see below) corresponding to the desired network (from scratch template or pre-trained model). For each keyword, two alternatives are possible:
    
 1. `None`: a generic `CNN` will be created, with random weights
 2. The path to pre-trained models (under `.pth` format) to be used in the analysis

Extra keywords can be added as column's headers in the `strategy.txt` file: they are needed for comparison purposes and/or to test the discriminator performances.


Training and testing are performed by alternative running `./src/aae_drive_bbfl.py` (for `broadband` and `filtered`) and `./src/aae_drive_hb.py` (for `hybrid`). The latter requires pre-trained `broadband` and `filtered` `DCAAE`: sequential training is possible (via `actions.txt` file) or pre-trained models can be adopted instead.

### Signal Databases

To train/test the different `DCAAE`, an extraction of 100 signals from the [STEAD database](https://github.com/smousavi05/STEAD/) is provided in the `database` folder. Seismic signals are 40.96 s-long, sampled at 100 Hz (`nt`=4096 time steps).

 - `ths_trn_nt4096_ls128_nzf8_nzd32.pth`: training set (80%)
 - `ths_tst_nt4096_ls128_nzf8_nzd32.pth`: testing set  (10%)
 - `ths_vld_nt4096_ls128_nzf8_nzd32.pth`: validation set (10%)

`ls` : latent space vector size
`nzd`: latent space channels (`broadband`)
`nzf`: latent space channels (`filtered`)

<p align="center">
  <img src="MRD_eqk_scatter.png" width="350" height="233" title="Figure 1: Hypocentral distance, magnitude and depth distribution of the earthquake sources">
</p>

The tag `nt4096_ls128_nzf8_nzd32.pth` is passed as input to the drive files `--dataset`. `--dataroot` flag indicates where the files with this tag are stored. 

## Train DCAAE

In the following, basics command line examples are provided to train each `DCAAE` (`broadband`,`filtered` and `hybrid`) over 5000 epochs (`--niter`) and with `--cuda`, over 1 GPU (`--ngpu`).

 - `broadband`:
 ```
    python3 ./src/aae_drive_bbfl.py --dataroot='./database/stead' --dataset='nt4096_ls128_nzf8_nzd32.pth' --cutoff=1. --imageSize=4096 --latentSize=128  --niter=5000 --cuda --ngpu=1 --nzd=32 --rlr=0.0001 --glr=0.0001 --outf='./imgs' --workers=8 --nsy=100 --batchSize=10 --actions='./actions_bb.txt' --strategy='./strategy_bb.txt'
 ```

<p align="center">
  <img src="aae_bb_example.png" width="350" height="500" title="Figure 2: Example of reconstructed broadband signal">
</p>


 - `filtered`:
 ```
python3 ./src/aae_drive_bbfl.py --dataroot='./' --dataset='nt4096_ls128_nzf8_nzd32.pth'  --cutoff=1. --imageSize=4096 --latentSize=128  --niter=5000 --cuda --ngpu=1 --nzf=8 --rlr=0.0001 --glr=0.0001 --outf='./imgs' --workers=8 --nsy=100 --batchSize=100 --actions='./actions_fl.txt' --strategy='./strategy_fl.txt' 
 ```

<p align="center">
  <img src="aae_fl_example.png" width="350" height="500" title="Figure 2: Example of reconstructed filtered signal">
</p>


 - `hybrid`
 ```
python3 ./src/aae_drive_hb.py --dataroot='./database/stead' --dataset='nt4096_ls128_nzf8_nzd32.pth' --cutoff=1. --imageSize=4096 --latentSize=128  --niter=3000 --cuda --ngpu=1 --nzd=32 --rlr=0.0001 --glr=0.0001 --outf='./imgs' --workers=8 --nsy=100 --batchSize=10 --actions='./actions_hb.txt' --strategy='./strategy_hb.txt'

 ```

<p align="center">
  <img src="aae_hb_example.png" width="350" height="500" title="Figure 3: Example of reconstructed hybrid signal">
</p>

