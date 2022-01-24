# image_caption
Captioning images using a CNN and Transformers. This repo is an attempt to implement captioning models from the paper titled "VirTex: Learning Visual Representations from Textual Annotations" (https://arxiv.org/abs/2006.06666) by Desai et.al.
The paper uses image captioning as a pretraining task for other downstream tasks, but this repo is centered around the task of generating dense captions for images. Everything used in
repo is written from scratch except the visual backbones (Pytorch pretrained models are used). Code for transformers is heavily inspired and even copied at places from the brilliant blog
post "The Annotated Transformer" (https://nlp.seas.harvard.edu/2018/04/03/attention.html). The other parts of code are inspired and taken in some places from notebooks for fast.ai course - Part2 (https://www.fast.ai/) 

## Some results on images from validation dataset:

![](images/res_1.jpg)


![](images/res_2.jpg)
