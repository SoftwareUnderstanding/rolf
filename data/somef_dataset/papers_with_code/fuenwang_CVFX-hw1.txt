# Homework1 Report

## Overview
The goal of this assignment is to train a color/texture transfer model using CycleGAN.

## Datasets
In this assignment, we use the `iphone2dslr_flower` dataset, which is introduced in the CycleGAN paper. `iphone2dslr_flower` contains iPhone and DSLR photos of flowers downloaded from Flickr photos. The main difference between these two classes is the depth of field in the images. Depth of field (DOF) is the distance between the nearest and the furthest objects that are in acceptably sharp focus in an image. We summarize the total number of examples and corresponding DOF type of each class under the below table. 

class | iPhone | DSLR |
--- | ---    | ---  |
DOF | Shallow  | Deep |
examples | 1813   | 3316 |
instance | <img src="output/imgs/sample_iphone.png" alt="drawing" width="150"/> | <img src="output/imgs/sample_dslr.png" alt="drawing" width="150"/> |

## About CycleGAN
In CycleGAN, they introduce two cycle consistency losses that capture the intuition that if
we translate from one domain to the other and back again we should arrive at where we started.
![](output/imgs/cycle_loss.png)
More specifically, the two losses are `forward cycle-consistency loss` and `backward cycle-consistency loss`.
1. forward cycle-consistency loss: x → G(x) → F(G(x)) ≈ x
2. backward cycle-consistency loss: y → F(y) → G(F(y)) ≈ y

In the original paper, the `total cycle loss` is defined as:
```python
loss = self.lambda1*forward_loss + self.lambda2*backward_loss
```
which is the sum of two L1 normalized loss(forward_loss amd backward_loss).

Apart from the `cycle loss`, the `identity loss` is also introduced in the paper. The intuition behind the `identity loss` is to encourage the mapping to preserve color composition between the input and output. 

##  Qualitative Results
We use our personal image (e.g., photoed by our iPhones) as inputs. As the table shown below, our model can learn to generate photos in the style of both DSLR and iPhone.

| DSLR ← iPhone | iPhone ← DSLR |
|---------------|---------------|
|<img src="output/imgs/inference/3.png" alt="drawing" width="300"/>|<img src="output/imgs/inference/6.png" alt="drawing" width="300"/>|
|<img src="output/imgs/inference/7.png" alt="drawing" width="300"/>|<img src="output/imgs/inference/1.png" alt="drawing" width="300"/>|
|<img src="output/imgs/inference/8.png" alt="drawing" width="300"/>|<img src="output/imgs/inference/2.png" alt="drawing" width="300"/>|
|<img src="output/imgs/inference/10.png" alt="drawing" width="300"/>|<img src="output/imgs/inference/4.png" alt="drawing" width="300"/>|
|<img src="output/imgs/inference/11.png" alt="drawing" width="300"/>|<img src="output/imgs/inference/5.png" alt="drawing" width="300"/>|
|<img src="output/imgs/inference/12.png" alt="drawing" width="300"/>|<img src="output/imgs/inference/9.png" alt="drawing" width="300"/>|

##  Comparison with Conventional Method
We compare our result with [*Color Transfer between Images*](http://www.thegooch.org/Publications/PDFs/ColorTransfer.pdf) [Reinhard et al., 2001]. In the paper, the authors utilize the L*a*b* color space and the mean and std of each L*, a*, and b* channel, respectively, to transfer the color between two images. We use the opencv implementation provided in [here](https://github.com/jrosebr1/color_transfer).

| Source (iPhone) | Target (DSLR) | Results |
|---------------|---------------|---------------|
|<img src="output/imgs/inference/a.jpg" alt="drawing" height="300"/>|<img src="output/imgs/inference/b.jpg" alt="drawing" height="300"/>|<img src="output/imgs/inference/c.jpg" alt="drawing" height="300"/>|

| Source (DSLR) | Target (iPhone) | Results |
|---------------|---------------|---------------|
|<img src="output/imgs/inference/b.jpg" alt="drawing" height="300"/>|<img src="output/imgs/inference/a.jpg" alt="drawing" height="300"/>|<img src="output/imgs/inference/d.jpg" alt="drawing" height="300"/>|

Our CycleGAN model shows significantly better results comparing to color transfer. Unlike color transfer which learns to transfer the style of a single selected piece of art, our CycleGAN model learns to mimic the style of an entire collection of dataset.

##  Appendix: Training Progress Bar
<img src="output/imgs/inference/bar.png" alt="drawing"/>
