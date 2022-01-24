# MAIC2021_Sleep

SNUH Medical AI Challenge 2021

Sleep AI challenge

SNUH x OUAR LAB X MBRC X NIA X MNC.AI

Teamname : SleepingDragon 

Crews: Hyeonhoon Lee, Si Young Yie, MinSeok Hong, SeungHoon Lee


1. Packages list

- numpy, pandas 
- sklearn, pytorch
- timm, torchvision
- cv2, matplotlib, PIL, Fmix, scipy, adamp, albumentations
- datetime, glob, time, math,  random 
- skimage, os, multiprocessing, tqdm, sys

2. Data preprocessing
- We tried image segmentation by each signals and trained the model with some meaningful segments,
  but the performance was not good. 
- Data imbalance was identified. But we thought that the degree of imbalance was not much severe. 
- Although upsampling might help the better performance of model, we decided not to do that.
- Because, the number of data is too large for the given computing environment. It takes much time to train 1 epoch (about 1.5hr to 2.0hr)
- However, some augmentation methods were used as follows: 
    1) Fmix (a variant of cutMix and MixUp) by recent published article (https://arxiv.org/abs/2002.12047)
    2) Coarsedropout and cutout by the package called albumentations

 
3. Modelling

- Model Architecture: Vision Transformer(ViT), Efficientnet-b4
- Tool: python, Pytorch
- Ensemble method: Use the weighted average of the square root of probability of each classes in models

4. Training
- Loss function: categorical cross entropy with label smoothing
- Training method: Automatic Mixed Precision(AMP)
- Optimizer: Adam for Vision Transformer, Adamp for Efficientnet-b4 
- Learning rate scheduler: CosineAnnealingWarmRestarts 
- Time for training: 1.5hr for 1 epoch

5. Postprocessing
- Our team include the MD specialist in Psychiatry. 
- In case of human scorers, the sleep stage is determined by the “context” of sleep in addition to the waveform of the graph.
- We made the post-processing algorithm take this contextual factor into account.

6. References
- https://arxiv.org/abs/2002.12047 
- https://arxiv.org/abs/2010.11929 
- https://developer.nvidia.com/blog/mixed-precision-training-deep-neural-networks/ 
- https://github.com/clovaai/AdamP 
- https://arxiv.org/pdf/1905.11946
