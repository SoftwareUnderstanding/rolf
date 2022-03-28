# DCGAN_pytorch
This repository is a simple Pytorch Implementation of DCGAN https://arxiv.org/abs/1511.06434  for generation of human faces using CelebA dataset.

Requriements:
Pytorch,
PIL,
Matplotlib,
Pickle

Dataset:
Download CelebA images from https://www.kaggle.com/jessicali9530/celeba-dataset 

To Train the model:
1. Update the dataset path 'train_data_path' in config.py
2. Run Train.py

To Generate images:
1. Update 'model_dir and 'model_path' of the trained model in 'run_inference.py'
2. Run run_inference.py
