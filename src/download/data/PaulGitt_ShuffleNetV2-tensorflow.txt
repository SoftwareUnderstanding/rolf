# ShuffleNetV2-tensorflow

ShuffleNetV2 from the paper: https://arxiv.org/abs/1807.11164.

This is not a official implementation.

System: Ubuntu 16.04, python3.5, tensorflow-1.14.0. tflearn-3.2.0

We use tensroflow and tflearn in a simple way to construct 
the ShuffleNetV2.

Using the oxford 17 catergories flowers dataset.
Because its image size fit the papper.
But there is a different part, we found the final output size
with convolutionaloperation not like the papper 7x7,
is 6x6, so we change a little bit.
