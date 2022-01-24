# HyperStackNet: A Hyper Stacked Hourglass Deep Convolutional Neural Network Architecture for Joint Player and Stick Pose Estimation in Hockey

This is the training pipeline used for:


H. Neher, K. Vats, A. Wong, D. A. Clausi,
**HyperStackNet: A Hyper Stacked Hourglass Deep Convolutional Neural Network Architecture for Joint Player and Stick Pose Estimation in Hockey**, CRV, 2018.

To run this code, make sure the following are installed:

- [Torch7](https://github.com/torch/torch7)
- hdf5
- cudnn

## Getting Started ##

Download the full [MPII Human Pose dataset](http://human-pose.mpi-inf.mpg.de), and place the `images` directory in `data/mpii`. From there, it is as simple as running `th main.lua -expID test-run` (the experiment ID is arbitrary). To run on [FLIC](http://bensapp.github.io/flic-dataset.html), again place the images in a directory `data/flic/images` then call `th main.lua -dataset flic -expID test-run`.

Most of the command line options are pretty self-explanatory, and can be found in `src/opts.lua`. The `-expID` option will be used to save important information in a directory like `pose-hg-train/exp/mpii/test-run`. This directory will include snapshots of the trained model, training/validations logs with loss and accuracy information, and details of the options set for that particular experiment.


## Acknowledgements ##

Code was modified from:

Alejandro Newell, Kaiyu Yang, and Jia Deng,
**Stacked Hourglass Networks for Human Pose Estimation**,
[arXiv:1603.06937](http://arxiv.org/abs/1603.06937), 2016.
