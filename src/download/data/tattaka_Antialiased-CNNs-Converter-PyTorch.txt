# Antialiased-CNN-Converter-PyTorch
Unofficial Pytorch module converter implementation of the paper Antialiased-CNN(https://arxiv.org/abs/1904.11486).

## Example
### Using `convert_model` function
``` python
import torch
from torch import nn
import numpy as np
import torchvision
from antialiased_cnns_converter import convert_model
m = torchvision.models.resnet18(True)
m = nn.DataParallel(m)
m = convert_model(m)
x = np.zeros((3, 3, 384, 576), dtype="f")
x = torch.from_numpy(x)
y = m(x)
print(y.size()) # Output:torch.Size([3, 1000])
```


Distributed under **MIT License** (See LICENSE)
