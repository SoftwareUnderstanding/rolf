# Objax - RepVGG: Making VGG-style ConvNets Great Again

An Objax (https://github.com/google/objax) implementation of RepVGG models based on the official PyTorch implementation (https://github.com/DingXiaoH/RepVGG)

Weights for models can be found here: https://drive.google.com/drive/folders/1mW3qbCpe9CMe0MLx_0BLVBAC7GbI5G6z?usp=sharing

## Abstract (taken from https://arxiv.org/abs/2101.03697)
We present a simple but powerful architecture of convolutional neural network, which has a VGG-like inferencetime body composed of nothing but a stack of 3 × 3 convolution and ReLU, while the training-time model has a
multi-branch topology. Such decoupling of the trainingtime and inference-time architecture is realized by a structural re-parameterization technique so that the model is
named RepVGG. On ImageNet, RepVGG reaches over 80%
top-1 accuracy, which is the first time for a plain model,
to the best of our knowledge. On NVIDIA 1080Ti GPU,
RepVGG models run 83% faster than ResNet-50 or 101%
faster than ResNet-101 with higher accuracy and show favorable accuracy-speed trade-off compared to the stateof-the-art models like EfficientNet and RegNet.

## Usage

To create a model with train architecture  
```python
from RepVGG import create_RepVGG_A0

train_model = create_RepVGG_A0(deploy = False)
```
To convert this model to inference / deploy architecture

```python
from RepVGG import convert

deploy_model = convert(model = train_model)
```

To create a model with inference / deploy architecture  
```python
from RepVGG import create_RepVGG_A0

model = create_RepVGG_A0(deploy = True)
```
To load pretrained weights 
```python
from RepVGG import create_RepVGG_A0, convert
import objax

model = create_RepVGG_A0(deploy = False)
objax.io.load_var_collection("path/to/RepVGG-A0-Train.npz", model.vars())
# do what you want with your train model
deploy_model = convert(model, save_path='RepVGG-A0-deploy.npz')
# do what you want with your deploy model
```
## JITing

JIT can be used in the normal way e.g. for inference 

```python
from RepVGG import create_RepVGG_A0
import objax
import objax.functional as F
from time import time


model = create_RepVGG_A0(deploy = True)


@objax.Function.with_vars(model.vars())
def predict(x):
    return F.softmax(model(x, training = False), axis = 1)

predict = objax.Jit(predict)

times = []

for i in range(5):
    test_input = objax.random.normal((10, 3, 200, 200))
    s = time()
    predict(test_input)
    e = time() - s
    times.append(e)

print(times) 
print(((sum(times[1:])/len(times[1:]) - times[0])/ times[0])*100)
# time per forward pass in s
# [3.19000506401062, 0.0007469654083251953, 0.0014159679412841797, 0.0007450580596923828, 0.0007431507110595703]
# -99.97138607896306
# gets you on average ~99.97% speed up (on cpu)
```

## Contribution 
Please feel free to raise PRs or issues to fix any bugs or to address any concerns 

## In Progress
Training examples will be provided soon

## Citations:
```bibtex

@ARTICLE{ding2021repvgg,
    title = {RepVGG: Making VGG-style ConvNets Great Again},
    author={Ding, Xiaohan and Zhang, Xiangyu and Ma, Ningning and Han, Jungong and Ding, Guiguang and Sun, Jian},
    journal={arXiv preprint arXiv:2101.03697},
    year={2021}
}

@software{objax-repvgg
    titile = {Objax-RepVGG},
    author = {Benjamin Ellis},
    url    = {https://github.com/benjaminjellis/Objax-RepVGG},
    year   = {2021}
}
```

