# Surgeon Action Detection for endoscopic images/videos
### This is a baseline model developed for SARAS-ESAD 2020 challenge. To download the dataset and participate in the challenge, please register at [SARAS-ESAD website](https://saras-esad.grand-challenge.org).

The code is adopated from [RetinaNet implementation in pytorch.1.x](https://github.com/gurkirt/RetinaNet.pytorch.1.x).

## Features of this baseline

- Data preparation instructions for SARAS-ESAD 2020 challenge
- Dataloader for SARAS-ESAD dataset
- Pytorch1.X implementation
- Feature pyramid network (FPN) architecture with different ResNet backbones
- Three types of loss functions i.e OHEM Loss, Focal Loss, and YOLO Loss on top of FPN

## Introduction

Here, we implement basic data-handling tools for [SARAS-ESAD](https://saras-esad.grand-challenge.org/Dataset/) dataset with FPN training process. We implement a pure pytorch code for train FPN with [Focal-Loss](https://arxiv.org/pdf/1708.02002.pdf) or [OHEM/multi-box-loss](https://arxiv.org/pdf/1512.02325.pdf) paper. 
<!-- Aim of this repository try different loss functions and make a fair comparison in terms of performance on SARAR-ESAD dataset. -->

We hope this will help kick start more teams to get up to the speed and allow the time for more innovative solutions. We want to eliminate the pain of building data handling and training process from scratch. Our final aim is to get this repository the level of [realtime-action-detection](https://github.com/gurkirt/realtime-action-detection).

At the moment we support the latest pytorch and ubuntu with Anaconda distribution of python. Tested on a single machine with 2/4/8 GPUs.

You can found out about architecture and loss function on parent repository, i.e. [RetinaNet implementation in pytorch.1.x](https://github.com/gurkirt/RetinaNet.pytorch.1.x).

ResNet is used as a backbone network (a) to build the pyramid features (b). 
Each classification (c) and regression (d) subnet is made of 4 convolutional layers and finally a convolutional layer to predict the class scores and bounding box coordinated respectively.

Similar to the original paper, we freeze the batch normalisation layers of ResNet based backbone networks. Also, few initial layers are also frozen, see `fbn` flag in training arguments. 

## Loss functions 
- OHEM with multi-box loss function: We use multi-box loss function with online hard example mining (OHEM), similar to [SSD](https://arxiv.org/pdf/1512.02325.pdf). A huge thanks to Max DeGroot, Ellis Brown for [Pytorch implementation](https://github.com/amdegroot/ssd.pytorch) of SSD and loss function.

- Focal loss: Same as in the original paper we use sigmoid focal loss, see [RetinaNet](https://arxiv.org/pdf/1708.02002.pdf). We use pure pytorch implementation of it.

- Yolo Loss: Multi-part loss function from [YOLO](https://pjreddie.com/darknet/yolo/) is also implemented here.

## Installation
You will need the following to run this code successfully
- Anaconda python
- Pytorch latest
- Visualisation 
  - if you want to visualise set tensorboard flag equal to true while training
  - [TensorboardX](https://github.com/lanpa/tensorboardX)
  - Tensorflow for tensorboard


### Datasets and other downloads
- Please visit [SARAS-ESAD](https://saras-esad.grand-challenge.org) website to download the dataset for surgeon action detection. 
- Extract all the sets (train and val) from zip files and put them under a single directory. Provide the path of that directory as data_root in train file. Data preprocessing and feeding pipeline is in [detectionDatasets.py](https://github.com/Viveksbawa/SARAS-ESAD-baseline/blob/master/data/detectionDatasets.py) file.
- rename the data directory `esad`. 
- Your directory will look like
  - esad
    - train
      - set1
        - file.txt
        - file.jpg
        - ..
    - val
      - obj
        - file.txt
        - file.jpg
        - ..

- Now your dataset is ready, that is time to download imagenet pretrained weights for ResNet backbone models. 
- Weights are initialised with imagenet pretrained models, specify the path of pre-saved models, `model_dir` in `train.py`. Download them from [torchvision models](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py). After you have download weights, please rename then appropriately under `model_dir` e.g. resnet50 resen101 etc. from This is a requirement of the training process. 

## TRAINING

Once you have pre-processed the dataset, then you are ready to train your networks.
We must have the following arguments set correctly:
- `data_root` is base path upto `esad` directory e.g. `\home\gurkirt\`
- `save_root` is a base path where you want to store the checkpoints, training logs, tensorboard logs etc.
- `model_dir` is a path where ResNet backbone model weights are stored

To train run the following command. 

```
python train.py --loss_type=mbox --data_root=\home\gurkirt\ --tensoboard=true
```

It will use all the visible GPUs. 
You can append `CUDA_VISIBLE_DEVICES=<gpuids-comma-separated>` at the beginning of the above command to mask certain GPUs. We used 2 GPU machine to run our experiments.

Please check the arguments in `train.py` to adjust the training process to your liking.

### Some useful flags
- 

## Evaluation
Model is evaluated and saved after each `1000` iterations. 

mAP@0.25 is computed after every `500` iterations and at the end. You can change to your liking by specify it in `train.py` arguments.

You can evaluate and save the results in `text` file using `evaluate.py`. It follow the same arguments `train.py`.
By default it evaluate using the model store at `max_iters`, but you can change it any other snapshot/checkpoint.

```
python evaluate.py --loss_type=focal
```

This will dump a log file with **results(mAP)** on the validation set and as well as a **submission file**.

## Results
Here are the results on `esad` dataset.

Results of the baseline models with different loss function and input image sizes, where  backbone network fixed to ResNet50. **AP_10**, **AP_30**, **AP_50**, and **AP_mean** are presented on validation-set, while **Test-AP_mean** is computed based on test-set similar to **AP_mean**.

| Loss   | min dim  | AP_10    | AP_30   |  AP_50   | AP_MEAN  |   Test-AP_MEAN |
|----- |----- |:-------: |:--------:| :-----: | :-------:| :------: |
| Focal  |  200  |  33.8  |  17.7  |  6.6  |  19.4  |  15.7  | 
| Focal  |  400  |  35.9  |  19.4  |  8.0  |  21.1  | **16.1** | 
| Focal  |  600  |  29.2  |  17.6  |  8.7  |  18.5  |  14.0  | 
| Focal  |  800  |  31.9  |  20.1  |  8.7  |  20.2  |  12.4  | 
| OHEM  |  200  |  35.1  |  18.7  |  6.3  |  20.0  |  11.3  | 
| OHEM  |  400  |  33.9  |  19.2  |  7.4  |  20.2  |  13.6  | 
| OHEM  |  600  |  37.6  |  23.4  |  11.2  |  24.1  |  12.5  | 
| OHEM  | 800 | **36.8** | **24.3** | **12.2** | **24.4** | 12.3|

Results of the baseline models with different loss function, backbone networks, where input image size is fixed to 400. **AP_10**, **AP_30**, **AP_50**, and **AP_mean** are presented on validation-set, while **Test-AP_mean** is computed based on test-set similar to **AP_mean**.

| Loss   | Network | AP_10    | AP_30   |  AP_50   | AP_MEAN  |  Test-AP_MEAN |
|----- |----- |:-------: |:--------:| :-----: | :-------:| :------: |
| Focal | ResNet18 | 35.1 | 18.9 | 8.1 | 20.7 | **15.3** |
| OHEM | ResNet18 | 36.0 | **20.7** | 7.7 | **21.5** | 13.8 |
| Focal | ResNet34 | 34.6 | 18.9 | 6.4 | 19.9 | 14.3 |
| OHEM | ResNet34 | **36.7** | 20.4 | 7.1 | 21.4 | 13.8 |
| Focal | ResNet50 | 35.9 | 19.4 | **8.0** | 21.1 | 16.1 |
| OHEM | ResNet50 | 33.9 | 19.2 | 7.4 | 20.2 | 13.6 |
| Focal | ResNet101 | 32.5 | 17.2 | 6.1 | 18.6 | 14.0 |
| OHEM | ResNet101 | 36.6 | 20.1 | 7.4 | 21.3 | 12.3 |

Outputs from the lastest model (800 OHEM) are uploaded in the sample folder. These are generated using the same model (800 OHEM). See flag at line [evaluate.py 114](https://github.com/Viveksbawa/SARAS-ESAD-Baseline/blob/master/evaluate.py#L114) to select validation or testing set (which will be available on 10th June).

## Details
- Input image size (`height x width`)is `600x1067` or `800x1422`.
- Batch size is set to `16`, the learning rate of `0.01`.
- Weights for initial layers are frozen see `freezeupto` flag in `train.py`
- max number of iterations is set to 6000
- SGD is used for the optimisation
- initial learning rate is set to `0.01`
- learning rate is dropped by the factor of 10 after 5000 iterations
- Different training setting might result in better/same/worse performance
