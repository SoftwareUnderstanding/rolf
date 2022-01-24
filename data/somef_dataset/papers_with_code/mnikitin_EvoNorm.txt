# EvoNorm
Gluon implementation of EvoNorm: https://arxiv.org/abs/2004.02967

## CIFAR-10 experiments

### Usage
Example of training *resnet20_v2* with *EvoNorm-S0*:<br/>
```
python3 train_cifar10.py --mode hybrid --num-gpus 1 -j 8 --batch-size 128 --num-epochs 186 --lr 0.003 --lr-decay 0.1 --lr-decay-epoch 81,122 --wd 0.0001 --optimizer adam --random-crop --model cifar_resnet20_v2 --evonorm s0
```

### Results
TBA
