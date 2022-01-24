# octconv-chainer
Implementation of octave convolution in Chainer (https://arxiv.org/abs/1904.05049)

ResNet50 ImageNet training for 90 epochs:

| alpha         | validation accuracy | theoretical flop cost|
| ------------- | ------------------- |----------------------|
| (origin)      | 0.762               |    100%              |
| 0.25          | 0.762               |    67%               |
| 0.50          | 0.757               |    44%               |
| 0.75          | 0.749               |    30%               |

Pretrained models are in the `pretrained_models/` directory

To run the pretrained model for validation:
```bash
$ python eval_imagenet.py <ImageNet validation directory> --model=octresnet50 --alpha=0.25 --pretrained-model=pretrained_models/octresnet50_alpha25.npz --gpu=0
$ python eval_imagenet.py <ImageNet validation directory> --model=octresnet50 --alpha=0.50 --pretrained-model=pretrained_models/octresnet50_alpha50.npz --gpu=0
$ python eval_imagenet.py <ImageNet validation directory> --model=octresnet50 --alpha=0.75 --pretrained-model=pretrained_models/octresnet50_alpha75.npz --gpu=0
```

The imageNet dataset preparation follows examples in chainercv.
https://github.com/chainer/chainercv/tree/master/examples/classification#how-to-prepare-imagenet-dataset

---


See `chainer-cifar10/` for training with cifar10 dataset. (The resnet model with octave convolution is `chainer-cifar10/models/oct_resnet.py`.)

---


reference: https://github.com/d-li14/octconv.pytorch
