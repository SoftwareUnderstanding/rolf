# Hair Cell Unet (HcUnet)
###### A library of deep learning functions for analysis of confocal z-stacks of mouse cochlear hair cells written in pytorch.

### Quickstart Guide
Bare minimum code requirement for evaluation of an image. 
```python
from hcat.unet import Unet_Constructor as Unet
import torch.nn as nn
from hcat import dataloader, transforms as t


data = dataloader.stack(path='./Data',
                        joint_transforms=[t.to_float(), t.reshape()],
                        image_transforms=[t.normalize()],
                        )

model = Unet(image_dimmensions=2
             in_channels=4,
             out_channels=1,
             feature_sizes=[8,16,32,64,128],
             kernel={'conv1': (3, 3, 2), 'conv2': (3, 3, 1)},
             upsample_kernel=(2, 2, 2),
             max_pool_kernel=(2, 2, 1),
             upsample_stride=(2, 2, 1),
             dilation=1,
             groups=1).to('cpu')



image, mask, pwl = data[0]

out = model.forward(image.float())
```




## **unet.py**

### _class_ **Unet_Constructor**
```python
model = Unet_Constructor(conv_functions=(nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.BatchNorm3d),
                         in_channels=3,
                         out_channels=2,
                         feature_sizes=[32, 64, 128, 256, 512, 1024],
                         kernel=(3, 3),
                         upsample_kernel=(2, 2),
                         max_pool_kernel=(2, 2),
                         upsample_stride=(2, 2),
                         dilation=1,
                         groups=1,
                        )
```
* **image_dimmensions:** Number of image dimmensions your images have. Only supports 2d or 3d images. 
* **in_channels:** Number of color channels the input image has 
* **out_channels:** Number of output features (including background)
* **feature_sizes:** list of integers representing the feature sizes at each step of the Unet. Each feature size must be twice the size of the previous. 
* **kernel:** tuple of size 2 (or 3 for 3d) representing the kernel sizes of the convolution operations of Unet
* **upsample_kernel:** tuple of size 2 (or 3 for 3d) representing the kernel sizes of the transpose convolutions for unet.
* **max_pool_kernel:** tuple of size 2 (or 3 for 3d) representing the kernel sizes of the maximum pooling operations of Unet
* **upsample_stride:** tuple of size 2 (or 3 for 3d) representing the stride of the transpose convolutions for unet. If an int is passed, it is automatically inferred that the stride is identical for all cardinal directions. i.e. upsample_stride=2 is the same as upample_stride=(2,2)
* **dilation:** dialation of the transpose convolution kernels
* **groups:** Number of groups of filters to be learned. Must be divisible by the number of input channels
#### _fun_ **forward**
```python
model.forward(image: torch.Tensor(dtype=torch.float))
```
**image**: torch.Tensor of type _float_ with shape [B, C, X, Y, Z] 
* B: Batch Size
* C: Number of Channels as defined by variable _in_channels_
* X: Size of image in x dimension
* Y: Size of image in y dimension
* Z: Size of image in z dimension

**returns**: ouput mask of torch.Tensor of type _float_ with shape [B, M, X*, Y*, Z*]
* B: Batch Size
* M: Number of mask Channels as defined by variable _out_channels_
* X*: Size of mask in x dimension
* Y*: Size of mask in y dimension
* Z*: Size of mask in z dimension

In all cases X*, Y*, and Z* will be less than X, Y, and Z due to only valid convolutions being used in the forward pass. The output mask will contain gradients unless model.eval() is called first. 

#### _fun_ **save**
```python
model.save(filename: str)
```
* **filename**: filename by which to serialize model state to

This function serializes the state of the model as well as initialization parameters. The save model can be loaded with model.load(filename)
* **returns** None

#### _fun_ **load**
```python
model.load(filename: str, to_cuda=True)
```
* **filename** Filename which to load model from. 
* **to_cuda** If true attemts to load the model state to cuda. If cuda is not available will throw a warning and initalize on the cpu instead. 
* **returns** None

## **dataloader.py**

### _class_ **stack**

```python
import dataloader
from hcat import transforms
data = dataloader.stack(path:str
                        joint_transforms:list
                       	image_transforms:list
                        out_transforms = [transforms.to_tensor()]
                       )
```



* **path** A string containing the location of your data
* **joint_transforms** A list of transforms to be applied to the images, masks, and pixel weighting images
* **image_transforms** A list of transforms to be applied to only the image
* **out_transforms** An optional list of transforms to be applied to all files after the application of functions contained in joint_transforms and image_transforms. Defaults to a list containing a single transform: transforms.to_tensor(). 

The class: dataloader.stack is a convenient way to easily load data from a file and apply data augmentation transforms for training the deep learning model. By default it looks for groups of similarly named **16bit or float** tif files with different exensions holding data for your original image, mask, and pixel-wise-weight for loss (pwl) adjustment (as described in the original Unet paper for forcing the network to learn borders between cells). The exensions must be:

	* *.tif
	* *.mask.tif
	* *.pwl.tif

For example, the dataloader will group three files with the names: data.tif (z-stack image), data.mask.tif (manually segmented mask for data.tif) and data.pwl.tif (computed pixel-wise weight augmenting the loss function). When placed in the same folder described by the input variable **path**, these images can then be indexed in a similar fasion to indexing an array. 

```python
image, mask, pwl = data[0]
```

The image, mask and pwl will first be augmented by transforms passed via the **joint_transforms** argument. Next the image will be augmented by transforms passed via the **image_transforms** argument. Then, image, mask, pwl will finnally be augmented by tranforms passed via the **out_transforms** argument. 

## **utils.py**

### _fun_ pad_image_with_reflections

```python

from hcat import utils

out = utils.pad_image_with_reflections(image, pad_size=(30, 30, 6)):
```

**image**: image of type torch.tensor with shape [B, C, X, Y, Z]

* B: Batch Size (**NOTE**: Currently only works for batch size of 1)
* C: Number of Channels
* X: Size of image in x dimension
* Y: Size of image in y dimension
* Z: Size of image in z dimension

**pad_size**: tuple of length 3 of the total padding in the three cardinal directions where _pad_size_ = (pad_x, pad_y, pad_z). Padding is applied to all sides of the image. 

**out**: image of type torch.tensor with shape  [B, C, X+pad_size[0], Y+pad_size[1], Z+pad_size[2]]

This function reflects the sides of an image to apply coninuous padding on all sides of a three dimmensional image as described in the original unet paper. (Image from: https://arxiv.org/pdf/1505.04597.pdf)

![image-20200421155211380](/Users/chrisbuswinka/Library/Application Support/typora-user-images/image-20200421155211380.png)

