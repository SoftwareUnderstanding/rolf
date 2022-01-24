## Object recognition and computer vision 2020/2021 bird classification competition

### This is my submission for the data challenge of the MVA recvis 2020 class

This submission allowed me to finish 1st out of 167 participants.

The solution consist in finetuning a Vision Transformer (ViT) (https://github.com/google-research/vision_transformer) and finetuning it using a heavy augmentation pipeline. We also use the Augmix (https://arxiv.org/abs/1912.02781) data augmentation and regularization to make sure we don't overfit our model which yields the optimal performances.

We preprocess the data finding the birds in the images using the Detecron2 library (https://github.com/facebookresearch/detectron2).

Challenge link: https://www.kaggle.com/c/mva-recvis-2020

Class link: https://www.di.ens.fr/willow/teaching/recvis20/
<p align='center'><img src= 'bird.png'/></p>
<p align='center'>Bird detection to focus the frame on the birds and remove background biases</p>
