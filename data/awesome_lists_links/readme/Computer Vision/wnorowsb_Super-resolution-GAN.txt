# SRGAN
A PyTorch implementation of SRGAN based on CVPR 2017 paper 
[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802).

## Requirements
- [Anaconda](https://www.anaconda.com/download/)
- PyTorch
```
conda install pytorch torchvision -c pytorch
```
- opencv
```
conda install opencv
```
## Changes in my fork
I've used networks models from another repo - https://github.com/aitorzip/PyTorch-SRGAN.
Activation functions were replaced by Swish (https://arxiv.org/abs/1710.05941).
Photos are now normalized before being passed to the networks. I've also added some small fixes to training algorithm, e.g. solving memory managment bug or preventing from saving ridiculous number of testing photos.
During my research I've discovered surprisingly poor network's performance on unscalled images. This led me to exploration of donwscale algotihm's impact on nets' results. You can check how trained model works on photos downscaled by different intrepolations by using '''test_interp.py''' script. 

