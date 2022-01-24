## Pytorch implementation of Octave Convolution

This is an implementation of the paper [Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution](https://arxiv.org/abs/1904.05049). Works with version 1.0. 

I'm not getting better results than fastai's xresnet implementation, and I'm also substantially slower. In theory, the FLOPs should be less, so there is something with the implementation that isn't quite right. 

Files:
- octconv.py: Contains the new layers.
- octxresnet: Makes the new layers work with fastai's xresnet architecture.
- test_oct_layer: A bunch of tests, including speed comparisons.
- imagenette_experiments: Experiments using ImageNettte (10 easy classes from Imagenet)
- imagewoof_experiments: Experiments using ImageWoof (10 hard classes from Imagenet).
