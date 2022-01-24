# Stream Image Classification using DCGANs for data regeneration with torch

Paper based on the results produced by this code is under review (refused) for ICIP 2018
You need to install Torch (http://torch.ch/) framework in order to reuse this code. 

In proposed approach we use DCGANs (original paper: https://arxiv.org/abs/1511.06434; code: https://github.com/soumith/dcgan.torch) to replace real data in the online learning on data stream scenario, where some historical data classes can be missing, which results in catastrophic forgetting during the classifier training. Detailed description of the approach can be found in our paper (that was not published, so it cannot really be found=( ).

MNIST dataset:
Prepared train and testsets can be downloaded here: https://drive.google.com/open?id=1XsY-ybjXpiEvRCg-4CjJqUcSJRo_FH8J
untar the datasets and put the files into ./datasets/MNIST/t7/

Stream_MNIST.lua is the main script to run the code on MNIST (smaller, faster and better results =D).
Stream_LSUN.lua  runs experiments on LSUN, but works quite bad for now (generators quality?)...

LSUN dataset preparation: 

Getting the dataset
- git clone https://github.com/fyu/lsun.git
- cd lsun
- python2.7 download.py (will download 160 Gb of images in .lmdb format, more details on the dataset can be found at http://lsun.cs.princeton.edu/2017/) 

README is in progress, for any question please contact Andrey Besedin at andrey.besedin@cea.fr 
