# FID score in PyTorch

## Requirements:
- pytorch
- torchvision

## Usage
To compute the FID score between two datasets and get gradient for the first dataset, where images of each dataset are contained in an individual folder:
```
python ./fid_score.py path/to/dataset1 path/to/dataset2
```

#### Example
```
python ./fid_score.py cifar/dev1 cifar/dev2
```

### Using different layers for feature maps

In difference to the official implementation, you can choose to use a different feature layer of the Inception network instead of the default `pool3` layer. 
As the lower layer features still have spatial extent, the features are first global average pooled to a vector before estimating mean and covariance.

This might be useful if the datasets you want to compare have less than the otherwise required 2048 images. 
Note that this changes the magnitude of the FID score and you can not compare them against scores calculated on another dimensionality. 
The resulting scores might also no longer correlate with visual quality.

You can select the dimensionality of features to use with the flag `--dims N`, where N is the dimensionality of features. 
The choices are:
- 64:   first max pooling features
- 192:  second max pooling featurs
- 768:  pre-aux classifier features
- 2048: final average pooling features (this is the default)

## Disclaimer

This implementation is heavily based on [this](https://github.com/mseitzer/pytorch-fid)

## License

This implementation is licensed under the Apache License 2.0.

FID was introduced by Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler and Sepp Hochreiter in "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", see [https://arxiv.org/abs/1706.08500](https://arxiv.org/abs/1706.08500)

The original implementation is by the Institute of Bioinformatics, JKU Linz, licensed under the Apache License 2.0.
See [https://github.com/bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR).
