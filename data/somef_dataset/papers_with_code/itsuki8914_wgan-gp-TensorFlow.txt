# wgan-gp-TensorFlow
Implemented wgan-gp using Tensorflow
see also:
https://github.com/ysasaki6023/faceGAN/tree/v1.0

https://github.com/changwoolee/WGAN-GP-tensorflow

I refered the above codes.

original paper:https://arxiv.org/abs/1704.00028

## usage
put named "data" folder in this directory.

data folder includes images for train.

to train

```
python main.py
```


to generate

```
python pred.py
```

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

