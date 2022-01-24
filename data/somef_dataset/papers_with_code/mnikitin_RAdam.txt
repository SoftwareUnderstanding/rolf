# RAdam
MXNet implementation of RAdam optimizer from [On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265) paper.

## Train CIFAR-10 using RAdam

```bash
python3 train_cifar10.py --mode hybrid \
                         --num-gpus 1 -j 8 \
                         --batch-size 128 \
                         --num-epochs 186 \
                         --lr 0.003 \
                         --lr-decay 0.1 \
                         --lr-decay-epoch 81,122 \
                         --wd 0.0001 \
                         --optimizer radam \
                         --model cifar_resnet20_v1
```
