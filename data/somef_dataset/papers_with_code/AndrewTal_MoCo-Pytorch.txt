## MoCo: Momentum Contrast for Unsupervised Visual Representation Learning

<p align="center">
  <img src="MoCo.png" width="300">
</p>

Modify from: https://github.com/facebookresearch/moco

### Unsupervised Training

#### Training

Use Multi-GPU with Pytorch: 
```
# Training
$ python train.py --image_folder /path/to/your/images

# Ckpt convert
$ python ckpt_convert.py --ckpt_path *.ckpt --save_path *.pth --arch resnet*
```
To run MoCo v1, set `--mlp False --moco-t 0.07 --cos False`.

## Citation

This is a PyTorch implementation of the [MoCo paper](https://arxiv.org/abs/1911.05722):

```
@Article{he2019moco,
  author  = {Kaiming He and Haoqi Fan and Yuxin Wu and Saining Xie and Ross Girshick},
  title   = {Momentum Contrast for Unsupervised Visual Representation Learning},
  journal = {arXiv preprint arXiv:1911.05722},
  year    = {2019},
}
```

It also includes the implementation of the [MoCo v2 paper](https://arxiv.org/abs/2003.04297):

```
@Article{chen2020mocov2,
  author  = {Xinlei Chen and Haoqi Fan and Ross Girshick and Kaiming He},
  title   = {Improved Baselines with Momentum Contrastive Learning},
  journal = {arXiv preprint arXiv:2003.04297},
  year    = {2020},
}
```


### 