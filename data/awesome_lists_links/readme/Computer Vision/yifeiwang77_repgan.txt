# GAN Sampling with REPGAN

Official code of [Reparameterized Sampling for Generative Adversarial Networks](https://arxiv.org/pdf/2107.00352) 
[ECML-PKDD 2021 Best ML paper]

Authors: Yifei Wang, Yisen Wang, Jiansheng Yang, and Zhouchen Lin (Peking University)

**What for: boostrapping the sample quality of pretrained GANs with latent-sample coupling MCMC**

---
## Install

Install dependencies with ```pip install -r requirements.txt```

## Usage

### 1. GAN Sampling with REPGAN

To draw samples with REPGAN, we load a pretrained GAN on CIFAR-10 and run the code

```
python main.py --dataroot [dataroot] --load-g [generator filename] --load-d [discriminator filename] --calibrate --num-images 50000
```
It will generate 50,000 images and save them in the numpy format. 

Notice: here we take [DCGAN](https://arxiv.org/pdf/1511.06434) in ```dcgan.py``` for an example. Other architectures (including WGAN) can also be adapted to fit our algorithm as it is model agnostic.

### 2. Customized Sampling

To intergrate REPGAN in your code, you can directly use / modify  the ```repgan``` function in ```repgan.py```, where it takes GANs as input and return a batch of samples. Detailed descriptions:

```
def repgan(netG, 
        netD, 
        calibrator, 
        device, 
        nz=100,
        batch_size=100, 
        clen=640, 
        tau=0.01, 
        eta=0.1):
    '''
    1) network config
    netG: generator network. Input: latent (B x latent_dim x 1 x 1). Output: images (B x C x H x W)
    netD: discriminator network. Input: images (B x C x H x W). Output: raw score (B x 1)
    calibrator: calibrator network for calibrating the discriminator score. Input: raw score (B x 1). Ouput: calibrated score: (B x 1)
    nz: the dimension of the latent z of the generator
    2) sampling config
    batch_size: number of samples per batch
    clen: length the Markov chain (only the last sample at the end of the chain is left)
    tau: step size in L2MC
    eta: scale of  white noise in L2MC. Default: sqrt(tau)
    3) update rule
    - (a) Langevin. z' = zk + tau/2 * grad + eta * epsilon
    - (b) MH test. Calculate alpha, and flip a coin with probability alpha.
    '''
```

---
If you find this codebase helpful, please cite 

```
@inproceedings{wang2021reparameterized,
  title={Reparameterized Sampling for Generative Adversarial Networks},
  author={Wang, Yifei and Wang, Yisen and Yang, Jiansheng and Lin, Zhouchen},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  pages={494--509},
  year={2021},
  organization={Springer}
}
```


## Reference
DCGAN example https://github.com/pytorch/examples/blob/master/dcgan/main.py

MHGAN code https://github.com/uber-research/metropolis-hastings-gans