# 2018-dlcv-team2
DLCV 2018 Team 2

In this project we used the architecture of Pix2Pix, which is a Conditional GAN, to colourise facades of buildings. Then we tried to transfer this learning to be able to colourise cats instead.

A link to the paper describing Pix2Pix can be found here:
https://arxiv.org/pdf/1611.07004v1.pdf

You can also find the repository in which we based the project in the folowing link: 
https://github.com/mrzhu-cool/pix2pix-pytorch

If you want to run the code yourself this is what you need installed:


    Linux
    Python with numpy
    NVIDIA GPU + CUDA 8.0 + CuDNNv5.1
    pytorch
    torchvision

Then you should clone the repo and extract the datasets. If you want to train the network you should run:
    
    python train.py --dataset facades --nEpochs 200 --cuda
  
And for training through transfer of the colourising of facades to cats:

    python train_transfer.py --dataset cat_dataset --nEpochs 50 --cuda
    
If you want to see the results the commands are 
  
    python test.py --dataset facades --model checkpoint/facades/netG_model_epoch_200.pth --cuda
    python test.py --dataset cat_dataset --model checkpoint/facades/netG_transfer_epoch_50.pth --cuda
    
A result is shown in the Jupyter notebook
