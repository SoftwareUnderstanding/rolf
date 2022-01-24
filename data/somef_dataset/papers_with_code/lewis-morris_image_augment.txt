# Basic image augmenter
Basic image augmentation for CNN's. Aiming to add regularisation to your model at the input layer.

Can be used in a pytorch pipeline 


## Setup

Use pip to install 

`pip install image-augment`

[PyPI](https://pypi.org/project/image-augment/)

## Features

Cutout - blocking out random segments of the image - details on this technique can be found here. https://arxiv.org/pdf/1708.04552.pdf

Noise - randomly adds noise to image 

Mirroring - randomly flips image on vertical and horizontal axis.

Rotation - randomly rotates image

## Basic Usage.


```python
from ImageAugment import basic
from PIL import Image

img = Image.open(./example_images/messi5.jpg)
img 
```

![messi](./example_images/messi5.jpg "Messi")


```python
new_image = basic()(img)
new_image
```

![messi](./example_images/messi_changed.jpg "Messi")

# For more info check the example jupyter notebook

[Jupyter examples](./example.ipynb)

