# SimpleSqueezeNet

This repo is simple C++ implementation of SqueezeNet [1] Convolutional Neural Network. Implemented code is capable of loading Caffe [2] pre-trained model parameters from HDF5 format and do forward pass. 

## Current status:

- early development, not tested!
- currently developed on Windows for MSVC, but should be multiplatform

## Requirements:

- [HDF5 library](https://www.hdfgroup.org/downloads/hdf5/)

## Howto:

- get and build HDF5 library
- extract HDF5 library to `./tools/HDF5`
- use batch file or run cmake directly 

## Refreences:
- [[1] IANDOLA, Forrest N., et al. Squeezenet: Alexnet-level accuracy with 50x fewer parameters and< 0.5 mb model size. arXiv preprint arXiv:1602.07360, 2016.](https://arxiv.org/abs/1602.07360)
- [[2] JIA, Yangqing, et al. Caffe: Convolutional architecture for fast feature embedding. In: Proceedings of the 22nd ACM international conference on Multimedia. ACM, 2014. p. 675-678.](https://dl.acm.org/citation.cfm?id=2654889)

## Licence:

- GPL v3