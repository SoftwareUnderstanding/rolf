# ResNet-PyTorch

### Update (Feb 20, 2020)

The update is for ease of use and deployment.

 * [Example: Export to ONNX](#example-export-to-onnx)
 * [Example: Extract features](#example-feature-extraction)
 * [Example: Visual](#example-visual)

It is also now incredibly simple to load a pretrained model with a new number of classes for transfer learning:

```python
from resnet_pytorch import ResNet 
model = ResNet.from_pretrained('resnet18', num_classes=10)
```

### Update (February 2, 2020)

This update allows you to use NVIDIA's Apex tool for accelerated training. By default choice `hybrid training precision` + `dynamic loss amplified` version, if you need to learn more and details about `apex` tools, please visit https://github.com/NVIDIA/apex.

### Overview
This repository contains an op-for-op PyTorch reimplementation of [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385).

The goal of this implementation is to be simple, highly extensible, and easy to integrate into your own projects. This implementation is a work in progress -- new features are currently being implemented.  

At the moment, you can easily:  
 * Load pretrained ResNet models 
 * Use ResNet models for classification or feature extraction 

_Upcoming features_: In the next few days, you will be able to:
 * Quickly finetune an ResNet on your own dataset
 * Export ResNet models for production
 
### Table of contents
1. [About ResNet](#about-resnet)
2. [Installation](#installation)
3. [Usage](#usage)
    * [Load pretrained models](#loading-pretrained-models)
    * [Example: Classify](#example-classification)
    * [Example: Extract features](#example-feature-extraction)
    * [Example: Export to ONNX](#example-export-to-onnx)
    * [Example: Visual](#example-visual)
4. [Contributing](#contributing) 

### About ResNet

If you're new to ResNets, here is an explanation straight from the official PyTorch implementation: 

Resnet models were proposed in "Deep Residual Learning for Image Recognition". Here we have the 5 versions of resnet models, 
which contains 5, 34, 50, 101, 152 layers respectively. Detailed model architectures can be found in Table 1. 

### Installation

Install from pypi:
```bash
$ pip3 install resnet_pytorch
```

Install from source:
```bash
$ git clone https://github.com/Lornatang/ResNet-PyTorch.git
$ cd ResNet-PyTorch
$ pip3 install -e .
``` 

### Usage

#### Loading pretrained models

Load an resnet18 network:
```python
from resnet_pytorch import ResNet
model = ResNet.from_name("resnet18")
```

Load a pretrained resnet18: 
```python
from resnet_pytorch import ResNet
model = ResNet.from_pretrained("resnet18")
```

Their 1-crop error rates on imagenet dataset with pretrained models are listed below.

| Model structure | Top-1 error | Top-5 error |
| --------------- | ----------- | ----------- |
|  resnet18       | 30.24       | 10.92       |
|  resnet34       | 26.70       | 8.58        |
|  resnet50       | 23.85       | 7.13        |
|  resnet101      | 22.63       | 6.44        |
|  resnet152      | 21.69       | 5.94        |

*Option B of resnet-18/34/50/101/152 only uses projections to increase dimensions.*

For results extending to the cifar10 dataset, see `examples/cifar`

#### Example: Classification

We assume that in your current directory, there is a `img.jpg` file and a `labels_map.txt` file (ImageNet class names). These are both included in `examples/simple`. 

All pre-trained models expect input images normalized in the same way,
i.e. mini-batches of 3-channel RGB images of shape `(3 x H x W)`, where `H` and `W` are expected to be at least `224`.
The images have to be loaded in to a range of `[0, 1]` and then normalized using `mean = [0.485, 0.456, 0.406]`
and `std = [0.229, 0.224, 0.225]`.

Here's a sample execution.

```python
import json

import torch
import torchvision.transforms as transforms
from PIL import Image

from resnet_pytorch import ResNet 

# Open image
input_image = Image.open("img.jpg")

# Preprocess image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

# Load class names
labels_map = json.load(open("labels_map.txt"))
labels_map = [labels_map[str(i)] for i in range(1000)]

# Classify with ResNet18
model = ResNet.from_pretrained("resnet18")
model.eval()

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to("cuda")
    model.to("cuda")

with torch.no_grad():
    logits = model(input_batch)
preds = torch.topk(logits, k=5).indices.squeeze(0).tolist()

print("-----")
for idx in preds:
    label = labels_map[idx]
    prob = torch.softmax(logits, dim=1)[0, idx].item()
    print(f"{label:<75} ({prob * 100:.2f}%)")
```

#### Example: Feature Extraction 

You can easily extract features with `model.extract_features`:
```python
import torch
from resnet_pytorch import ResNet 
model = ResNet.from_pretrained('resnet18')

# ... image preprocessing as in the classification example ...
inputs = torch.randn(1, 3, 224, 224)
print(inputs.shape) # torch.Size([1, 3, 224, 224])

features = model.extract_features(inputs)
print(features.shape) # torch.Size([1, 512, 1, 1])
```

#### Example: Export to ONNX  

Exporting to ONNX for deploying to production is now simple: 
```python
import torch 
from resnet_pytorch import ResNet 

model = ResNet.from_pretrained('resnet18')
dummy_input = torch.randn(16, 3, 224, 224)

torch.onnx.export(model, dummy_input, "demo.onnx", verbose=True)
```

#### Example: Visual

```text
cd $REPO$/framework
sh start.sh
```

Then open the browser and type in the browser address [http://127.0.0.1:10004/](http://127.0.0.1:10004/).

Enjoy it.

#### ImageNet

See `examples/imagenet` for details about evaluating on ImageNet.

### Credit

#### Deep Residual Learning for Image Recognition

*Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun*

##### Abstract

Deeper neural networks are more difficult to train. We
present a residual learning framework to ease the training
of networks that are substantially deeper than those used
previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual
networks are easier to optimize, and can gain accuracy from
considerably increased depth. On the ImageNet dataset we
evaluate residual nets with a depth of up to 152 layers—8×
deeper than VGG nets [41] but still having lower complexity. An ensemble of these residual nets achieves 3.57% error
on the ImageNet test set. This result won the 1st place on the
ILSVRC 2015 classification task. We also present analysis
on CIFAR-10 with 100 and 1000 layers.
The depth of representations is of central importance
for many visual recognition tasks. Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset. Deep
residual nets are foundations of our submissions to ILSVRC
& COCO 2015 competitions1
, where we also won the 1st
places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation.

[paper](http://arxiv.org/abs/1512.03385) [code](https://github.com/KaimingHe/deep-residual-networks)

```text
@article{He2015,
	author = {Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
	title = {Deep Residual Learning for Image Recognition},
	journal = {arXiv preprint arXiv:1512.03385},
	year = {2015}
}
```