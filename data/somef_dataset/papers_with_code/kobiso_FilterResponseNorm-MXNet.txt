# FilterResponseNorm-MXNet

MXNet implementation of [Filter Response Normalization Layer (FRN)](https://arxiv.org/abs/1911.09737) published in CVPR2020

<p align="center">
  <img src="figures/performance.png" width="500" />
</p>
<p align="center">
  <img src="figures/frn.png" width="500" />
</p>

## Features
- 1D(NxCxW), 2D(NxCxHxW), 3D(NxCxDxHxW) FilterResponseNorm
- Learnable epsilon parameter

## How to use

### Prerequisites
- Python 3.x
- MXNet

### Usage example

```python

from frn import FilterResponseNorm1d, FilterResponseNorm2d, FilterResponseNorm3d

class Net(gluon.Block):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(20, kernel_size=(5,5))
        self.frn1 = FilterResponseNorm2d(num_features=20, epsilon=1e-6, is_eps_learnable=False,
                 tau_initializer='zeros', beta_initializer='zeros', gamma_initializer='ones')
        self.avg_pool = nn.GlobalAvgPool2D()
        self.frn2 = FilterResponseNorm1d(num_features=10, epsilon=1e-6, is_eps_learnable=False,
                 tau_initializer='zeros', beta_initializer='zeros', gamma_initializer='ones')
        self.fc2 = nn.Dense(10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.frn1(x)
        x = F.relu(x)
        x = self.avg_pool(x)
        x = self.frn2(x)
        x = self.fc2(x)

        return x
```

## Reference
- Paper: [Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks](https://arxiv.org/abs/1911.09737)
- Repository: [Filter Response Normalization Layer in PyTorch](https://github.com/gupta-abhay/pytorch-frn)

## Author
Byungsoo Ko / kobiso62@gmail.com
