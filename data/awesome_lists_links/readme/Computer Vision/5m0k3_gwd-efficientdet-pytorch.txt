# pytorch based EfficientDet solution - Global Wheat Detection
A complete pytorch pipeline for training, cross-validation and inference notebooks used in Kaggle competition [Global Wheat Detection](https://www.kaggle.com/c/global-wheat-detection) (May-Aug 2020)

## Table of Contents

- [Brief overview of the competition images](#Brief-overview-of-the-competition-images)
- [Notebooks description](#Notebooks-description)
  - [[TRAIN] notebook](#TRAIN-notebook)
  - [[CV] Cross Validation notebook](#CV-Cross-Validation-notebook)
  - [[INFERENCE] Submission notebook](#INFERENCE-Submission-notebook)
- [How to use](#How-to-use)
- [Improvements](#Improvements)

## Brief overview of the competition images
Wheat heads were from various sources:  
<a href="https://imgur.com/HhOQtba"><img src="https://imgur.com/HhOQtba.jpg" title="head" alt="head" /></a>  
A few labeled images are as shown: (Blue bounding boxes)  
<a href="https://imgur.com/QhnuEEf"><img src="https://imgur.com/QhnuEEf.jpg" title="head" alt="head" width="378" height="378" /></a> <a href="https://imgur.com/5yUJCPV"><img src="https://imgur.com/5yUJCPV.jpg" title="head" alt="head" width="378" height="378" /></a>  

## Notebooks description
A brief content description is provided here, for detailed descriptions check the notebook comments  

### [TRAIN] notebook
  1. Pre-Processing:  
    - Handled the noisy labels (too big/small boxes etc.)  
    - Stratified 5 fold split based on source  
    
  2. Augmentations:  
    - Albumentations - RandomSizedCrop, HueSaturationValue, RandomBrightnessContrast, RandomRotate90, Flip, Cutout, ShiftScaleRotate  
    - **Mixup** - https://arxiv.org/pdf/1710.09412.pdf  
      2 images are mixed  
    <a href="https://imgur.com/HkDFQ2g"><img src="https://imgur.com/HkDFQ2g.jpg" title="head" alt="head" /></a>  
    - **Mosaic** - https://arxiv.org/pdf/2004.12432.pdf  
      4 images are cropped and stitched together  
    <a href="https://imgur.com/YZn47iN"><img src="https://imgur.com/YZn47iN.jpg" title="head" alt="head" width="378" height="378" /></a>  
    - Mixup-Mosaic: Combining the above two, applying mixup to 2 (top-right and bottom-left) of the 4 quarters of mosaic  
    <a href="https://imgur.com/py9WrZn"><img src="https://imgur.com/py9WrZn.jpg" title="head" alt="head" width="378" height="378" /></a>  
    
  3. Configurations:  
    - Optimizer - Adam Weight Decay (AdamW)  
    - LR Scheduler - ReduceLRonPleateau (initial LR = 0.0003, factor = 0.5)  
    - Model - EfficientDet D5 (pytorch implementation of the original tensorflow version)  
    - Input Size - 1024 * 1024  
    - Last and Best 3 checkpoints saved  
    
### [CV] Cross Validation notebook
  1. Pre-Processing:  
    - Same as in [TRAIN]  
    
  2. Test Time Augmentations:  
    - Flips and Rotate  
    <a href="https://imgur.com/kSZlHWr"><img src="https://imgur.com/kSZlHWr.jpg" title="head" alt="head" /></a>  
    - Color shift  
    - Scale  (scale down with padding)  
    
  3. Ensemble:  
    - Support for ensembling of multiple folds of the same model  
    - [Weighted Boxes Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion) is used to ensemble final predicted boxes  
    
  4. Automated Threshold Calculations:  
    - Confidence level threshold is calculated based on ground truth labels  
    - Optimal Final CV score (Metric: IoU) is obtained through this  
    
### [INFERENCE] Submission notebook
  1. Test Time Augmentations:  
    - Same as in [CV]  
    
  2. Pseudo Labelling:  
    - Multi-Round Pseudo Labelling pipeline based on https://arxiv.org/pdf/1908.02983.pdf  
    - Implemented Cross Validation calculations at the end of each round to decide the best thresholds for Pseudo Labels in the next round  
    - Training pipeline same as in [TRAIN]  
    <a href="https://imgur.com/qqbI8zE"><img src="https://imgur.com/qqbI8zE.jpg" title="head" alt="head" width="512" height="512" /></a>  
    
  3. Post-Processing and Result:  
    - Included final bounding boxes reshaping function  
    (Red : Original | Blue : Altered {+5%})  
    <a href="https://imgur.com/SOyNX40"><img src="https://imgur.com/SOyNX40.jpg" title="head" alt="head" width="378" height="378" /></a>  
    - Final predictions made with ensembled combinations of TTA  
    
## How to use
Just change the directories according to your environment.  

Google Colab deployed versions are available for  
**[TRAIN]** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qpKUsWzwebEyTZcOMHS15i5k9xzcRgXg?usp=sharing)  
**[CV]** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XZtml9hbHxCMdO9zbyzfJ1IecQpND_Pt?usp=sharing)  

In case of any deprecation issues/warnings in future, use the modules available in Resources folder.  

## Improvements
Acknowledging the shortcomings is the first step for progress. Thus, listing the possible improvements that could've made my Model better:  
  - Ensemble Multi-Model/Fold predictions for Pseudo Labels, currently single model is used to make pseudo labels. Would've made the model more robust to noise too.  
  - GAN or Style Transfer could've been used to produce more similar labeled images from the current train images for better generalization.
  - Relabeling of noisy labels using multi-folds. (Tried but failed)
  - IoU loss used in training should be replaced by modern SOTA GIoU, CIoU or DIoU
    
