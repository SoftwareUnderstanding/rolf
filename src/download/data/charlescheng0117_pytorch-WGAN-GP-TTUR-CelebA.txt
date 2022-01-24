## WGAN-GP-TTUR on CelebA
An PyTorch implementation of WGAN with gradient penalty and TTUR.
- WGAN: https://arxiv.org/abs/1701.07875
- Improved Training of Wasserstein GANs (gradient penalty): https://arxiv.org/abs/1704.00028
- GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (TTUR): https://arxiv.org/abs/1706.08500

Environment
---
- OS: Ubuntu16.04
- Language: Python
- Packages: torch, torchvision, numpy, tensorflow (for tensorboard) 

Prepare Data
---
1. Download CelebA from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html (the Google Drive link)
    - Unzip them, and find the zip file with the name `img_align_celeba.zip`, which contains the training data to use
2. Create a folder: `mkdir data/celeba/`
3. Unzip the zip file: `unzip img_align_celeba.zip` and put the zipped files under the folder we just created: `data/celeba/`

Training
---
Run the following for training
```
python main.py --dataset celeba --dataroot data/celeba --batch_size 64 --image_size 128 --niter 10000 --exp celeba_experiment
```
and check the log in tensorboard with
```
tensorboard --logdir .
```

Acknowledgement
---
Implementation is hugely borrowed from
1. https://github.com/martinarjovsky/WassersteinGAN
2. https://github.com/igul222/improved_wgan_training
3. https://github.com/bioinf-jku/TTUR
4. https://github.com/LynnHo/WGAN-GP-DRAGAN-Celeba-Pytorch
5. https://github.com/tensorflow/tensor2tensor
