# WGAN-GP-ResBlock-TensorFlow
Implementation of WGAN-GP-ResBlock-TensorFlow.

This implementation is based on https://github.com/itsuki8914/wgan-gp-TensorFlow .

Usual blocks are replaced by residual block.

It may more efficient than conventional one.

WGAN-GP original paper: https://github.com/changwoolee/WGAN-GP-tensorflow

Referenced Network architecture: https://arxiv.org/pdf/1802.05637.pdf


## usage
put named "data" folder in this directory.

data folder includes images for train.

like this
```
main.py
pred.py
data
  ├ 000.jpg
  ├ aaa.png
  ...
  └ zzz.jpg
```

to train

```
python main.py
```


to generate

```
python pred.py
```


## Result examples

Generation of anime faces

<img src = 'Result/example.png' width = '600px'>
