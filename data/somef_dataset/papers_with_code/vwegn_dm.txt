# DeepMatching

This code implements the DeepMatching algorithm as a fully trainable deep neural network in TensorFlow using the Keras framework as proposed in [[1]](#1). It includes the usage of a pre-trained VGG network for feature extraction [[5]](#5).

This work was created as part of a Master Thesis in the Institute for Applied Mathematics in the Department of Mathematics and Informatics of WWU Münster.


## Installation

__Info:__ As of 2020 the following installation process does not support GPU execution of the code anymore. Currently, there are no plans for updates to TensorFlow 2 or other versions.


Create virtual environment and activate:
+ (```conda config --append channels conda-forge```)
+ ```conda create -n <env> python=3.6```
+ ```conda activate <env>```

Manual package installation:
+ ```conda install -c conda-forge tensorflow=1.8.0```
+ ```conda install -c conda-forge keras=2.1.6```
+ ```conda install -c conda-forge scikit-image```



Automatic package installation via ```requirements.txt```:
+ ```conda install --file requirements.txt```

Get the Sintel dataset for training or matching evaluation:
+ Download the training sequences from [[3]](#3) and extract them to the `/MPI-Sintel/training` folder.


## Usage
This implementation provides three different modes of operation:
+ `dm_match_pair` computes matches for a pair of images based on given weights (`weights.h5py`). These matches are then visualized as optical flow via `flow_vis` [[4]](#4).
+ `dm_match` computes matches based on given weights for a set of testing images and evaluates the matching results by comparing with the ground truth.
+ `dm_train` trains the neural network. 

The sequences on which training and the matching evaluation are performed are loaded from `MPI-Sintel/training/clean`. The path can be customized in the function `read_training_data()`  in `dm.py`. Weights are saved to and loaded from `weights.h5py`.

### Configuration
All modes of operation run on a configuration specified by the following parameters as introduced in [[2]](#2).
+ `alpha` is the stride of subsampling.
+ `beta` is the offset of subsampling.
+ `radius` defines a window in which matches are computed.
+ `levels` sets the amount of levels in the pyramid hierarchy of the DeepMatching algorithm.

The set default configuration (for testing purposes) is `alpha = 8, beta = 4, radius = 10, levels = 3`. In [[2]](#2) the configuration is set to `alpha = 8, beta = 4, radius = 80, levels = 6`, but due to limited resources it was not possible to train with this configuration.


### Running the Code
An example for the `dm_match_pair` method is provided in `example.py`.

## Experiments & Results
The picture below is an example of the matching results. It was performed with the following configuration: `alpha = 8, beta = 4, radius = 40, levels = 3`. 

Note, that as opposed to `dm_match` in `dm_match_pair` occlusions and out of radius pixels are not removed, which leads to more errors and therefore some artifacts in the picture. Also, no further optical flow optimization is applied. Due to subsampling the resolution of the resulting image is low.


![](/images/example_alley_1.png)

### Evaluation
|  | Untrained | Trained* | Difference |
| -------- | -------- | -------- | -------- |
| Acc@2     |  79.11%    |   79.24%   |  0.14%    |
| Acc@5     |  85.09%    |   85.28%   |   0.18%   |
| Acc@10     |  87.32%    |    87.45%  |   0.12%   |

*Training was performed on 8 Sintel training sequences (clean) over 50 epochs (executed on a NVIDIA TITAN V with 12GB memory). Configuration: `alpha = 8, beta = 4, radius = 40, levels = 3`. Training includes fine-tuning of the VGG network.

It can be seen that an approximate improvement of `0.1%` was achieved. This coincides with the results in [[2]](#2) and leaves room for further improvements (e.g. through _reciprocal verification_).



## References
* <a id="1">[1]</a> Jerome Revaud et al. “DeepMatching: Hierarchical Deformable Dense Matching”. URL: https://arxiv.org/pdf/1506.07656
* <a id="2">[2]</a> James Thewlis et al. “Fully-Trainable Deep Matching”. URL: http://arxiv.org/abs/1609.03532
* <a id="3">[3]</a> Sintel Dataset: http://sintel.is.tue.mpg.de/
* <a id="4">[4]</a> Optical Flow Visualization: https://github.com/tomrunia/OpticalFlow_Visualization
* <a id="5">[5]</a> Karen Simonyan, Andrew Zisserman. “Very Deep Convolutional Networks for Large-Scale Image Recognition“. URL: https://arxiv.org/abs/1409.1556



