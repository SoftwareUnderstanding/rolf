# PointNet1_Keras
> Implementation of PointNet-1 on Point Cloud segmentation problem by Keras(based Tensorflow).

This repo is implementation for PointNet(https://arxiv.org/abs/1612.00593) in Keras. The model is in pointnet.py.

Original tensorflow implementation: https://github.com/charlesq34/pointnet

Pytorch implementation: https://github.com/fxia22/pointnet.pytorch

This code was built and developed based on this repo: https://github.com/garyli1019/pointnet-keras

![](outputs/segmented_airplane.png)

## Installation

### Anaconda/Miniconda

Windows & OS X & Linux:

```sh
conda env create -f environment.yml
```

## Usage 

Below is guide to download data, train network then test prediction on Shapenet dataset

1. Download and unzip the [Shapenet dataset](https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_v0.zip) to `./pointnet1_keras/DATA` directory
1. Run training, type on terminal: `python train_segmentation.py`
1. Run testing, type on terminal: `python test_segmentation.py`

## Meta

NGUYEN CONG MINH – [@Facebook](https://www.facebook.com/minhnc.social) – minhnc.edu.tw@gmail.com

Distributed under the MIT license. See ``LICENSE`` for more information.

[https://github.com/minhncedutw/pointnet1_keras](https://github.com/minhncedutw/pointnet1_keras)

## Contributing

1. Fork it (<https://github.com/yourname/yourproject/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

#### [README Template](https://github.com/dbader/readme-template) from [Dan Bader](https://dbader.org/blog/write-a-great-readme-for-your-github-project)
