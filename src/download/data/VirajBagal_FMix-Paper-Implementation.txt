# FMix-Paper-Implementation
In this repo, I have implemented Fmix, Cutmix and Mixup on the Fashion MNIST dataset using PreAct ResNet18. This is an attempt to reprdouce the results from this paper:  https://arxiv.org/abs/2002.12047

Link to my medium blog: https://medium.com/@virajbagal12/mixed-sample-data-augmentation-721e65093fcf

```python
python train.py --msda baseline --save_dir path_to_save
```

# Baseline

![Screenshot](Baseline/fashionmnist.png)

# Mixup

![Screenshot](Mixup/mixup_fashionmnist.png)

# Cutmix

![Screenshot](Cutmix/cutmix_fashionmnist.png)

# Fmix

![Screenshot](Fmix/masks.png)


![Screenshot](Fmix/fmix_noaug.png)

# Accuracy Curve Comparison

![Screenshot](acc_fmix_new.png)

# Original vs Reproduced

![Screenshot](fmix_bar_graph.png)
