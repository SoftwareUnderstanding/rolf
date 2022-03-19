# VisTrans
Implementations of transformers based models for different vision tasks

## Install
1) Install from <a href="https://pypi.org/project/vistrans/">PyPI</a>

```Python
pip install vistrans
```
2) Install from <a href="https://anaconda.org/nachiket273/vistrans">Anaconda</a>

```Python
conda install -c nachiket273 vistrans
```

## Version 0.003 (06/30/2021)
------------------------------
[![PyPI version](https://badge.fury.io/py/vistrans.svg)](https://badge.fury.io/py/vistrans)

Minor fixes to fix issues with existing models.


## Version 0.002 (04/17/2021)
------------------------------
[![PyPI version](https://badge.fury.io/py/vistrans.svg)](https://badge.fury.io/py/vistrans)

Pretrained Pytorch <a href="https://arxiv.org/pdf/2101.11605v1.pdf">Bottleneck Transformers for Visual Recognition</a> including following
<br>
* botnet50
* botnet101
* botnet152
<br>
Implementation based off <a href="https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2">Official Tensorflow Implementation</a><br>

## Usage
---------------------
pip install vistrans
```
1) List Pretrained Models.
```Python
from vistrans import BotNet
BotNet.list_pretrained()
```
2) Create Pretrained Models.
```Python
from vistrans import BotNet
model = BotNet.create_pretrained(name, img_size, in_ch, num_classes,
                                 n_heads, pos_enc_type)
```
3) Create Custom Model
```Python
from vistrans import BotNet
model = BotNet.create_model(layers, img_size, in_ch, num_classes, groups,
                            norm_layer, n_heads, pos_enc_type)
```

## Version 0.001 (03/04/2021)
-----------------------------
[![PyPI version](https://badge.fury.io/py/vistrans.svg)](https://badge.fury.io/py/vistrans)

Pretrained Pytorch <a href="https://arxiv.org/abs/2010.11929">Vision Transformer</a> Models including following
<br>
* vit_s16_224
* vit_b16_224
* vit_b16_384
* vit_b32_384
* vit_l16_224
* vit_l16_384
* vit_l32_384
<br>
Implementation based off <a href=https://github.com/google-research/vision_transformer>official jax repository</a> and <a href="https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py">timm's implementation</a><br>

## Usage
---------------------
1) List Pretrained Models.
```Python
from vistrans import VisionTransformer
VisionTransformer.list_pretrained()
```
2) Create Pretrained Models.
```Python
from vistrans import VisionTransformer
model = VisionTransformer.create_pretrained(name, img_size, in_ch, num_classes)
```
3) Create Custom Model
```Python
from vistrans import VisionTransformer
model = VisionTransformer.create_model(img_size, patch_size, in_ch, num_classes,
                                       embed_dim, depth, num_heads, mlp_ratio,
                                       drop_rate, attention_drop_rate, hybrid,
                                       norm_layer, bias)
```
