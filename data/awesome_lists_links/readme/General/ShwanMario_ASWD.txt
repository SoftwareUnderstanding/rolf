# Augmented-Sliced-Wasserstein-Distances

This repository provides the code to reproduce the experimental results in the paper **Augmented Sliced Wasserstein Distances**.
## Prerequisites

### Python packages

To install the required python packages, run the following command:

```
pip install -r requirements.txt
```

### Datasets
Two datasets are used in this repository, namely the [CIFAR10](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.222.9220&rep=rep1&type=pdf) dataset and [CELEBA](http://openaccess.thecvf.com/content_iccv_2015/html/Liu_Deep_Learning_Face_ICCV_2015_paper.html) dataset.
- The CIFAR10 dataset (64x64 pixels) will be automatically downloaded from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz when running the experiment on CIFAR10 dataset. 
- The CELEBA dataset needs be be manually downloaded and can be found on the website http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html, we use the cropped CELEBA dataset with 64x64 pixels.

### Precalculated Statistics

To calculate the [Fréchet Inception Distance (FID score)](https://arxiv.org/abs/1706.08500), precalculated statistics for datasets

- [CIFAR 10](http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_cifar10_train.npz) (calculated on all training samples)
- [cropped CelebA](http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_celeba.npz) (64x64, calculated on all samples)

are provided at: http://bioinf.jku.at/research/ttur/.
## Project & Script Descriptions
Two experiments are included in this repository, where benchmarks are from the paper [Generalized Sliced Wasserstein Distances](http://papers.nips.cc/paper/8319-generalized-sliced-wasserstein-distances) and the paper [Distributional Sliced-Wasserstein and Applications to Generative Modeling](https://arxiv.org/pdf/2002.07367.pdf), respectively. The first one is on the task of sliced Wasserstein flow, and the second one is on generative modellings with GANs. For more details and setups, please refer to the original paper **Augmented Sliced Wasserstein Distances**.
### Directories
- ```./result/ASWD/CIFAR/``` contains generated imgaes trained with the ASWD on CIFAR10 dataset.
- ```./result/ASWD/CIFAR/fid/``` FID scores of generated imgaes trained with the ASWD on CIFAR10 dataset are saved in this folder.
- ```./result/CIFAR/``` model's weights and losses in the CIFAR10 experiment are stored in this directory.

Other setups follow the same naming rule.
### Scripts
The sliced Wasserstein flow example can be found in the [jupyter notebook](https://anonymous.4open.science/repository/e55153b2-70be-4089-9362-1443ddfaece4/Sliced%20Waaserstein%20Flow.ipynb).

The following scripts belong to the generative modelling example:
- [main.py](https://anonymous.4open.science/repository/e55153b2-70be-4089-9362-1443ddfaece4/main.py) : run this file to conduct experiments.
- [utils.py](https://anonymous.4open.science/repository/e55153b2-70be-4089-9362-1443ddfaece4/utils.py) : contains implementations of different sliced-based Wasserstein distances.
- [TransformNet.py](https://anonymous.4open.science/repository/e55153b2-70be-4089-9362-1443ddfaece4/TransformNet.py) : edit this file to modify architectures of neural networks used to map samples. 
- [experiments.py](https://anonymous.4open.science/repository/e55153b2-70be-4089-9362-1443ddfaece4/experiments.py) : functions for generating and saving randomly generated images.
- [DCGANAE.py](https://anonymous.4open.science/repository/e55153b2-70be-4089-9362-1443ddfaece4/DCGANAE.py) : neural network architectures and optimization objective for training GANs.
- [fid_score.py](https://anonymous.4open.science/repository/e55153b2-70be-4089-9362-1443ddfaece4/fid_score.py) : functions for calculating statistics (mean & covariance matrix) of distributions of images and the FID score between two distributions of images.
- [inception.py](https://anonymous.4open.science/repository/e55153b2-70be-4089-9362-1443ddfaece4/inception.py) : download the pretrained [InceptionV3](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.html) model and generate feature maps for FID evaluation.

## Experiment options for the generative modelling example
The generative modelling experiment evaluates the performances of GANs trained with different sliced-based Wasserstein metrics. To train and evaluate the model, run the following command:

```
python main.py  --model-type ASWD --dataset CIFAR --epochs 200 --num-projection 1000 --batch-size 512 --lr 0.0005
```

### Basic parameters
- ```--model-type``` type of sliced-based Wasserstein metric used in the experiment, available options: ASWD, DSWD, SWD, MSWD, GSWD. Must be specified.
- ```--dataset``` select from: CIFAR, CELEBA, default as CIFAR.
- ```--epochs``` training epochs, default as 200.
- ```--num-projection``` number of projections used in distance approximation, default as 1000.
- ```--batch-size``` batch size for one iteration, default as 512.
- ```--lr``` learning rate, default as 0.0005.

### Optional parameters

- ```--niter``` number of iteration, available for the ASWD, MSWD and DSWD, default as 5.
- ```--lam``` coefficient of regularization term, available for the ASWD and DSWD, default as 0.5.
- ```--r``` parameter in the circular defining function, available for GSWD, default as 1000.


## References 
### Code
The code of generative modelling example is based on the implementation of [DSWD](https://github.com/VinAIResearch/DSW) by [VinAI Research](https://github.com/VinAIResearch).

The pytorch code for calculating the FID score is from https://github.com/mseitzer/pytorch-fid.

### Papers
- [Distributional Sliced-Wasserstein and Applications to Generative Modeling](https://arxiv.org/pdf/2002.07367.pdf)
- [Generalized Sliced Wasserstein Distances](http://papers.nips.cc/paper/8319-generalized-sliced-wasserstein-distances)
- [Sliced Wasserstein Auto-Encoders](https://openreview.net/forum?id=H1xaJn05FQ)
- [Max-Sliced Wasserstein Distance and its Use for GANs](http://openaccess.thecvf.com/content_CVPR_2019/html/Deshpande_Max-Sliced_Wasserstein_Distance_and_Its_Use_for_GANs_CVPR_2019_paper.html)
