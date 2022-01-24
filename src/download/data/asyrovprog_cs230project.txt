# CS230: Building footprint extraction based on RGBD satellite imagery

![](doc/maskrcnn.png)

![](doc/condgan.png)

## Overview

In this project we explore instance and semantic segmentation using Mask R-CNN ([arxiv](https://arxiv.org/abs/1703.06870)) and Conditional Adversarial Networks ([arxiv](http://arxiv.org/abs/1611.07004)). Inputs are satellite images from Urban3D dataset ([github](https://spacenetchallenge.github.io/datasets/datasetHomePage.html)).

Writing:
http://cs230.stanford.edu/projects_winter_2020/reports/32060950.pdf

## Installation 

Dataset is about 30GB and can be downloaded from aws as described [here](doc/datasets.md). After downloading dataset install dependencies and preprocess dataset:

1. Follow instructions at https://github.com/matterport/Mask_RCNN to install Mask-RCNN model libraries from github 

2. Install project dependencies:

   ```bash
   pip3 install -r requirements.txt
   ```

3. Download pre-trained COCO weights (`mask_rcnn_coco.h5`) from the [releases page](https://github.com/matterport/Mask_RCNN/releases). We use them to do transfer learning. 

4. Copy `mask_rcnn_coco.h5` into `mask_rcnn` folder.

5. Setup environment variable DATASET_ROOT and make sure it exists. 

6. Run pre-processing of the dataset. Preprocessing splits every tile into set of smaller images and also splits original Test dataset into Dev and Test datasets.

   ```bash
   python3 build_dataset.py
   ```

## Configuration (Optional)

Optionally, review configuration files for the instance segmentation with Mask R-CNN are under /configs/ directory. The most important ones are configuration of baseline model (urban3d_baseline_rgb.py) and configuration of final optimal model (urban3d_optimal_rgbdt.py).



Semantic Segmentation with Conditional Adversarial Network does not config files.



## Instance Segmentation with Mask R-CNN

### Run Training

To train model from scratch using pre-trained weights, run the training script:
    ```bash
    python -m experiments.urban3d_training --config optimal_rgbdt
    ```

### Run Validation and Visualization

To run model inference on a few examples from test set and to compute precision/recall/F1 metrics on the entire test set, run validation script:
    ```bash
    python -m experiments.urban3d_validation --config optimal_rgbdt --dataset "test" 
    ```

To visualize model from inside/outside using occlusion maps and saliency maps, run visualization script:

    ```bash
    python -m experiments.urban3d_visualization --config optimal_rgbdt --dataset "test"
    ```



## Semantic Segmentation with Conditional Adversarial Network.

### Run Training

To train model from scratch using pre-trained weights, run the training script:

    ```bash
    python -m experiments.urban3d_training_cgan --itype "rgbd" --epochs 20
    ```

After this, _manually_ copy all files from `logs/urban3d_cond_gan_rgbd/models/` into `models/urban3d_cond_gan_rgbd/models/`

### Run Validation and Visualization

To run model inference on a few examples from dev set and to compute precision/recall/F1 metrics on the entire dev set, run validation script:

    ```bash
    python -m experiments.urban3d_validation_cgan --itype "rgbd" --dataset "test"
    ```

To compute precision/recall/F1/IoU metrics on the entire test set, run:

    ```bash
    python -m experiments.urban3d_visualization_cgan --itype "rgbd" --dataset "test"
    ```

Above command saves output in `models/urban3d_cond_gan_rgbd/images` folder.

