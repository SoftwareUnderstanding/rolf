# Spatial Broadcast Decoder
Spatial Broadcast Decoder Implementation using Tensorflow v1.12. This implements only the decoder as proposed in the paper.

See the original paper for specifications:

**Spatial Broadcast Decoder: A Simple Architecture for Learning Disentangled Representations in VAEs**
https://arxiv.org/abs/1901.07017

Note that the convolutional part of the spatial broadcast decoder has a different implementation than in the paper. Just go ahead and adjust the channel numbers for each layer. Kernel and stride are original.
