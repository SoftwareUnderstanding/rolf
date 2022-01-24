# SRGAN

## Verification environment

- Python 3.9.6
- Tensorflow 2.6.0

## Network structure

![https://arxiv.org/pdf/1609.04802.pdf](https://github.com/Hayashi-Yudai/SRGAN/blob/main/assets/network_img.png "cited from https://arxiv.org/pdf/1609.04802.pdf")
cited from https://arxiv.org/pdf/1609.04802.pdf

## How to train

### Prepare your dataset

Make the following directory tree for your dataset on the project root and place original images in `train/high_resolution` and `validate/high_resolution/` directories.

```
.datasets
└── (your dataset name)
    ├── test
    │   ├── high_resolution
    │   └── low_resolution
    ├── train
    │   ├── high_resolution
    │   └── low_resolution
    └── validate
        ├── high_resolution
        └── low_resolution
```

Next, make low resolution images which have quarter size of original ones and place them in `low_resolution` directories.

This program request TFRecords as dataset. I prepare a function for you. Fix `dataset_name` and `extension` in the src/datasets.py and execute it from project root.

```bash
python src/dataset.py
```

Make sure there exists `train.tfrecords` and `valid.tfrecords` in the `datasets/(your dataset name)` directory.

### Configure parameters

The parameters like hyper-parameters are set in the config.yaml

### Training

#### Pre-training SRResNet

config.yaml

```
TYPE: SRResNet
EPOCHS: 10000
BATCH_SIZE: 16
IMG_HEIGHT: 32
IMG_WIDTH: 32
LEARNING_RATE: 0.0001
TRAIN_DATA_PATH: ./datasets/train.tfrecords
VALIDATE_DATA_PATH: ./datasets/valid.tfrecords
CHECKPOINT_PATH: ./checkpoint/generator_train
START_EPOCH: 0
GEN_WEIGHT:
DISC_WEIGHT:
G_LOSS: 100000000
```

Start training with the following command

```bash
$ python src/train.py
$ pipenv run train  # If you use pipenv
```

#### Training SRGAN

config.yaml

```
TYPE: SRGAN
EPOCHS: 10000
BATCH_SIZE: 16
IMG_HEIGHT: 32
IMG_WIDTH: 32
LEARNING_RATE: 0.0001
TRAIN_DATA_PATH: ./datasets/train.tfrecords
VALIDATE_DATA_PATH: ./datasets/valid.tfrecords
CHECKPOINT_PATH: ./checkpoint/gan_train
START_EPOCH: 0
GEN_WEIGHT: ./generator_train/generator_best
DISC_WEIGHT:
G_LOSS: 100000000
```

Start training with the following command

```bash
$ python src/train.py
$ pipenv run train  # If you use pipenv
```

## Example

You can download pre-trained weight from [here](https://drive.google.com/drive/folders/1LE1AK0HVHN-_x9S3-aCUeA0mFT-D68VW?usp=sharing). These weights are trained with 32x32 images.

![](https://github.com/Hayashi-Yudai/SRGAN/blob/main/assets/test_data.png)
