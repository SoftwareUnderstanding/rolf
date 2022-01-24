# SupContrast.paddle
A PaddlePaddle implementation of SupContrast: Supervised Contrastive Learning

<p align="center">
  <img src="images/teaser.png" width="700">
</p>

This repo covers an reference implementation for the following papers in PaddlePaddle 2.x, using CIFAR as an illustrative example:  
(1) Supervised Contrastive Learning. [Paper](https://arxiv.org/abs/2004.11362)  
(2) A Simple Framework for Contrastive Learning of Visual Representations. [Paper](https://arxiv.org/abs/2002.05709)  

## Loss Function
The loss function [`SupConLoss`](https://github.com/paddorch/SupContrast.paddle/blob/main/src/models/supcon.py#L46) in `supcon.py` takes `features` (L2 normalized) and `labels` as input, and return the loss. If `labels` is `None` or not passed to the it, it degenerates to SimCLR.

Usage:
```python
from supcon import SupConLoss

# define loss with a temperature `temp`
criterion = SupConLoss(temperature=temp)

# features: [bsz * n_views, f_dim]
# `n_views` is the number of crops from each image
# better be L2 normalized in f_dim dimension
features = ...
# labels: [bsz]
labels = ...

# SupContrast
loss = criterion(features, labels)
# or SimCLR
loss = criterion(features)
...
```

## Comparison
Results on CIFAR-10:

|          |Arch | Setting | Loss | Paper Acc(%) | Our Acc(%) | abs. improv. |
|----------|:----:|:---:|:---:|:---:|:---:|:---:|
|  SupCrossEntropy | ResNet50 | Supervised   | Cross Entropy |  95.0  | **96.9** <br> (-*)    | 1.9 |
|  SupContrast     | ResNet50 | Supervised   | Contrastive   |  96.0  | **97.3** <br> (96.8*) | 1.3 |
|  SimCLR          | ResNet50 | Unsupervised | Contrastive   |  93.6  |    -     |  -  |

> *for no `cutout` Accuracy

## Running
You might use `CUDA_VISIBLE_DEVICES` to set proper number of GPUs.

We released 3 models, please download from [cowtransfer](https://cowtransfer.com/s/7b3ee056bbd042) with code `461254`:
```
./logs
|-- resnet50-ce-final/final             # SupCrossEntropy (Acc: 96.9)
|-- resnet50-supcon-final/final         # SupContrast Pretrained
|-- resnet50-linear-final/final         # SupContrast Linear Fine-tuned (Acc: 97.3)
```

**(0) Data Preparing**
```
cd data
wget https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz
```

**(1) Standard Cross-Entropy**
- Train:
```
python main_ce.py -y config/resnet50_ce.yml
```

- Test:

config the `continue_from` in `config/resnet50_ce.yml` to specify the checkpoint path, then run:
```
python main_ce.py -y config/resnet50_ce.yml --test
```

you will get:

![](images/ce_test.png)

**(2) Supervised Contrastive Learning**  

- Train:

Pretraining stage:
```
python main_supcon.py -y config/resnet50_supcon.yml
```

Linear evaluation stage:
config the `from_supcon` in `config/resnet50_linear.yml` to specify the checkpoint path, then run:

```
python main_ce.py -y config/resnet50_linear.yml
```

- Test:

config the `continue_from` in `config/resnet50_linear.yml` to specify the checkpoint path, then run:
```
python main_ce.py -y config/resnet50_linear.yml --test
```

you will get:

![](images/supcon_test.png)


## Details
- see `config/` for configuration details

### Differences
- Compared to the original batch size of 6144, we use a smaller batch size of 128, which allows us to train on a single GPU card.
- We use **gradient clip** to avoid gradient explosion, to make the training of small batch size more stable

### Data Augmentation
- AutoAugment: https://github.com/DeepVoltaire/AutoAugment
- Cutout: https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py

## Reference
```
@Article{khosla2020supervised,
    title   = {Supervised Contrastive Learning},
    author  = {Prannay Khosla and Piotr Teterwak and Chen Wang and Aaron Sarna and Yonglong Tian and Phillip Isola and Aaron Maschinot and Ce Liu and Dilip Krishnan},
    journal = {arXiv preprint arXiv:2004.11362},
    year    = {2020},
}
```
- https://github.com/HobbitLong/SupContrast
- https://github.com/itisianlee/paddle-cifar100
