# ISBI2012 Segmentation with PyTorch

<br>

## Basic Info

### Dataset

ISBI2012: <http://brainiac2.mit.edu/isbi_challenge/>

<br><br>

### Backbone Network

**U-Net: Convolutional Networks for Biomedical Image Segmentation** \<[arxiv link](https://arxiv.org/abs/1505.04597)\>

```
@misc{ronneberger2015unet,
    title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
    author={Olaf Ronneberger and Philipp Fischer and Thomas Brox},
    year={2015},
    eprint={1505.04597},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

<br><br>

## Data Preprocessing

- Download ISBI2012 Dataset from <http://brainiac2.mit.edu/isbi_challenge/>  
- Save data like below.  

```
# ROOT Directory

ISBI2012
├── train-volume.tif
├── train-labels.tif
└── test-volume.tif --> (Not Necessary)
```

- Split Data to train, val, test with [data_prep.ipynb](./data_prep.ipynb)

<br>

## Train

```bash
$ python train.py
```

<br>

## Test

```bash
$ python eval.py
```

Then, you can see result images from `./test_results`  

<br><br>

## Reference

- [hanyoseob - youtube-cnn-002-pytorch-unet](https://github.com/hanyoseob/youtube-cnn-002-pytorch-unet)
