# SegNet-keras-implementation
An implementation of SegNet(https://arxiv.org/abs/1511.00561) in keras

The repository doesn't contain dataset, please prepare and set up it in config.py. A nicely normalized and cleaned dataset can be downloaded from https://github.com/alexgkendall/SegNet-Tutorial.

To train a new model:

```console
python train.py --save [name of model to be saved]
```

To load a model and resume training:

```console
python train.py --save [name of model to be saved] --resume --load [name of model to be loaded]
```

To test a trained model

```console
python test.py --load [name of model to read]
```

