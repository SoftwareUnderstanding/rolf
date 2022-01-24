# Place classification using DeepLab-v2 with PyTorch <!-- omit in toc --> 

This is an implementation of place classification method of "Place classification based on semantic segmented objects".
### COCO-Stuff

Pre-trained Deeplab-v2 with COCO-stuff 164k dataset is available at <a href='https://drive.google.com/file/d/18kR928yl9Hz4xxuxnYgg7Hpi36hM8J2d/view?usp=sharing'>164k Model</a>.

## Setup

### Requirements

* Python 2.7+/3.6+
* Anaconda environment

Then setup from `conda_env.yaml`. Please modify cuda option as needed (default: `cudatoolkit=10.0`)

```console
$ conda env create -f configs/conda_env.yaml
$ conda activate deeplab-pytorch
```

### Datasets

Setup instruction is provided in each link.

* [COCO-Stuff 10k/164k](data/datasets/cocostuff/README.md)


### Building weight matrix

```
Usage: place_weight_matrix.py single -c ./configs/cocostuff164k.yaml -m ./models/deeplabv2_resnet101_msc-cocostuff164k-100000.pth -i .
```

### Place classification using built weight matrix
```
Usage: place_classification.py single -c ./configs/cocostuff164k.yaml -m ./models/deeplabv2_resnet101_msc-cocostuff164k-100000.pth -i .
```


## References
- DeepLab-v2 implementation of [kazuto1011](https://github.com/kazuto1011/deeplab-pytorch). 
