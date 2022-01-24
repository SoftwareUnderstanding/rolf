# Deep-Learning
Here densely connected convolutional network is applied on CIFAR10 dataset.
## Goal
The goal of this assignment was to successfully implement densely connected convolutional network and play with the hyperparamters and understand how each layer actually works.
## Implementation
For my program I used 2 densed block in the learning method. I followed some parts from the paper but not all of it. To run the program I used Google Colab and chose the runtime GPU as the dataset is huge.

## Results
My accuracy result is around 47% which is not so good. It could be because I did not use several layers in the dense blocks even implemented only two dense blocks.
If we use more Dense Blocks with multiple layers (as per the architecture provided in the paper) we might get better accuracy.

## Notes

I tried to implement more Dense Blocks with multiple layers but due to some limitations it was taking really long to generate the results and sometimes it was crashing. This is why I sticked to the simple form of the densely connected convolutional network architecture in order to clearly understand and demonstrate how it works.

## Acknowledgements
1. https://arxiv.org/pdf/1608.06993.pdf
