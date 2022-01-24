# Lambda Networks Implementation

<p align="center">
  <img src="https://neurohive.io/wp-content/uploads/2021/01/rsz_lambd.png" width="800">
</p>



This is the implementation of the paper: 

 ["**LambdaNetworks: Modeling Long-Range Interactions Without Attention**,"](https://arxiv.org/abs/2102.08602) I. Bello *et al*., 2021 <br>



## Required package ###
  - Python 3
  - PyTorch ,torchvision

## Usage

###  Train

```bash
python train.py --help

# An example of training script
python train.py \
--gpu 0 \
--lr 1e-2 \
--weight-decay 1e-4 \
--num-epoch 100 \
--batch-size 256
```
## Author ##

[@ Jae-Hyun Park](https://github.com/jaehyunnn)
