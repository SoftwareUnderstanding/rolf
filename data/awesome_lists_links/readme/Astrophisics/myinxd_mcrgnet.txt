# MCRGNet
<TODO>

## Methods
In this repository, we provide a toolbox namely `MCRGNet` to classify radio galaxies of different morphologies. Our method is designed based on the state-of-the-art Convolutional Neural Network (CNN), which is trained and applied under a three step framework as
1. Pretraining the network unsupervisedly with unlabeled samples (P).
2. Finetuing the pretrained network parameters supervisedly with labeled samples (F).
3. Classify a new radio galaxy by the trained network. (C).

## Installation
To utilize our toolbox on radio galaxy morphology classification, a convolutional autoencoder (CAE) network should be trained in advance and saved. Installation of our python based scripts is as follows, 

```sh
$ cd MCRGNet
$ <sudo> pip install <--user> . 
```

## Requirements
To run our scripts, some python packages are required, which are listed as follows.

- [numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/)
- [matplotlib](http://www.matplotlib.org)
- [astropy](http://docs.astropy.org/en/stable/)
- [requests](http://www.python-requests.org/en/master/)
- [Tensorflow](http://www.tensorflow.org)

The [requirements file](https://github.com/myinxd/MCRGNet/blob/master/requirements.txt) is provided in this repository, by which the required packages can be installed easily. We advice the users to configure these packages in a virtual environment.

- initialize env
```sh
$ <sudo> pip install virtualenv
$ cd MCRGNet
$ virtualenv ./env
```
- install required packages
```sh
$ cd MCRGNet
$ env/bin/pip install -r ./requirements.txt
``` 

In addition, the computation can be accelerated by paralledly processing with GPUs. In this work, our scripts are written under the guide of [Nvidia CUDA](https://developer.nvidia.com/cuda-downloads), thus the Nvidia GPU hardware is also required. You can either refer to the official guide to install CUDA, or refer to this brief [guide](https://github.com/myinxd/MCRGNet/blob/master/cuda_installation.md) by us.


## Demos and Usage
To use the MCRGNet, we provide demos to show how to pretrain and finetune the network. Note that the [jupyter-notebook](http://jupyter.org/) is required.

1. [cae-pretrain-demo](https://github.com/myinxd/MCRGNet/blob/master/demo/cae-pretrain-demo.ipynb): Pre-train the network
2. [cnn-finetune-demo](https://github.com/myinxd/MCRGNet/blob/master/demo/cae-pretrain-demo.ipynb): Fine-tune the network

In the toolbox, you can design your own CAE and CNN network of optional layers as well as parameters by the Class `ConvNet`, please refer to the script for details.

Some useful command-line-executable python scripts are also provided and archived in the utils folder,
- [dataDownload](https://github.com/myinxd/MCRGNet/blob/master/utils/dataDownload.py): Retrieve radio galaxy samples from the FIRST archive.
- [getEstLabel](https://github.com/myinxd/MCRGNet/blob/master/utils/getEstLabel): Get estimated label for the radio galaxies to be classified.

In addition, some new tools are on the way, cheers 

## References
- [TensorFlow tutorial](https://www.tensorflow.org/tutorials/)
- [Requests](http://docs.python-requests.org/en/master/)

## Author
- Zhixian MA <`zxma_sjtu(at)qq.com`>

## License
Unless otherwise declared:

- Codes developed are distributed under the [MIT license](https://opensource.org/licenses/mit-license.php);
- Documentations and products generated are distributed under the [Creative Commons Attribution 3.0 license](https://creativecommons.org/licenses/by/3.0/us/deed.en_US);
- Third-party codes and products used are distributed under their own licenses.

## Citation
Coming soon...
