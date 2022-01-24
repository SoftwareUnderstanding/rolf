# Lyft 3D Object Detection for Autonomous Vehicles
This project contains the source code for [Lyft 3D Object Detection for Autonomous Vehicles competition(Lyft 3D Object Detection for Autonomous Vehicles) on Kaggle.

## Trained Models
This folder contains the models we've trained and used.
1. UnetEnsemble15Epoch.pth - This model is used for U-Net Baseline and in the PSPNet-UNET Ensemble.
2. UnetEnsemble13Epoch.pth, UnetEnsemble14Epoch.pth, UnetEnsemble15Epoch.pth - These models are used in U-NET Ensemble

*Due to GitHub upload size restrictions, we could not upload the PSPNet trained model.  We tried compression, but the file size surpassed the upload limits.*

## util.py
This file contains many utility functions for working with the provided data set of LiDAR, camera images, and semantic maps.  Additionally, this file contains the BEV data loader to load pre-trained data to feed into networks.

## DataPreprocessing.ipynb
This notebook contains the code and logic for pre-processing the training data.  Since the data provided is in multiple forms: LiDAR, camera images, and semantic maps, we preprocess the data prior to training. We first transform the LiDAR point cloud from the sensor’s reference frame to the car’s reference frame. Next, we voxelize the LiDAR points to project them into a 3-dimensional space. Our final training image is a bird’s eye view (top down) projection of the world around the car. Figure 1 shows an example of this. During training, the data loader concatenates the bird’s eye image representation with the semantic map. (Note: This data processing is used by the Baseline U-Net model; we are still imple- menting preprocessing for the PSPNet model).

## BaselineUNetModel.ipynb
This notebook is our baseline model for the competition.  It is an implementation of the U-Net Model (Olaf, et. al. 2015). It loads the weights pre-trained by the author, and makes predictions based on validation dataset, which is split by train dataset with about 70/30 ratio. Then, the notebook generates a CSV file called `baseline_val_pred.csv` which fits the submission format of the competition.  

**Attribution:** This notebook is borrowed from https://www.kaggle.com/meaninglesslives/lyft3d-inference-prediction-visualization, and customized to for our environment.

### radam-optimizer.py
This is an efficient Adam optimizer which has a lower memory footprint, and allows us to train on the large dataset.

## PSPNet_ResNet.ipynb
This notebook is an implementation of the Pyramid Scene Parsing Network (Zhao, et. al. 2017).  In addition, it uses the ResNET pre-trained weights to achieve use transfer learning and achieve higher predictions.  This implementatino is based on Trusov's PSPNet model implementation (Trusov).

## PSPNet_ResNet_UNet_ensemble
This notebook is an implementation of the Ensemble of PSPNet and UNet model.  The PSPNet model uses ResNET pre-trained weights to use transfer learning.

## EvaluatePredictionAndGroundTruthScores.ipynb
This notebook compares the predict output and groud truth table and calculates the average score which is defined in the evaluation metrics in the report. Make sure that `baseline_val_pred.csv` and `val_gt.csv` exist and the paths to these two csv file are configured correctly.

## reference-model.ipynb
This is a notebook provided by the competition. We're mainly using this to understand the data aspect of the project.

## References
All data used in this competition is provided by Lyft here: https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/overview/.
```
@misc{rsnet2015,
    title={Deep Residual Learning for Image Recognition},
    author={Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
    year={2015},
    eprint={1512.03385},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
@article{UNETModel,
  author    = {Olaf Ronneberger and
               Philipp Fischer and
               Thomas Brox},
  title     = {U-Net: Convolutional Networks for Biomedical Image Segmentation},
  journal   = {CoRR},
  volume    = {abs/1505.04597},
  year      = {2015},
  url       = {http://arxiv.org/abs/1505.04597},
  archivePrefix = {arXiv},
  eprint    = {1505.04597},
  timestamp = {Mon, 13 Aug 2018 16:46:52 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/RonnebergerFB15},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
@misc{semseg2019,
  author={Trusov, Roman},
  title={pspnet-pytorch},
  howpublished={\url{https://github.com/Lextal/pspnet-pytorch}},
  year={2019}
}
@inproceedings{zhao2017pspnet,
  title={Pyramid Scene Parsing Network},
  author={Zhao, Hengshuang and Shi, Jianping and Qi, Xiaojuan and Wang, Xiaogang and Jia, Jiaya},
  booktitle={CVPR},
  year={2017}
}
```
