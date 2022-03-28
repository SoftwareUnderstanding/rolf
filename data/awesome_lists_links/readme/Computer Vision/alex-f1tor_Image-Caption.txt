# Image caption

This project #2 at the Udacity Computer Vision nanodegree:  generation of a picture description using an encoder-decoder deep learning architecture.

## Brief description:

  - This project based on original arxiv paper [Show and Tell: A Neural Image Caption Generator (2015)](https://arxiv.org/abs/1411.4555) paper;
  - The encoder is pretrained [resnet50](https://arxiv.org/abs/1512.03385) deep CNN [available](https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html#resnet50) in pyTorch; 
  - The caption generator was trained on [MS-COCO 2014 dataset](http://cocodataset.org/#download).
 
## Installation:

The following steps for default using pretrained model:
```sh 
$ git clone git clone https://github.com/alex-f1tor/Image-Caption.git
$ cd Image-Caption
$ mkdir models
$ cd models
$  wget https://drive.google.com/open?id=19mcr08t6gY0UcUiAKTkBPO8MP0_wsghV -O 'decoder-4.pkl' && wget https://drive.google.com/open?id=1xe4zTMQAnH8QxcwHF7-i2lnmoBecJPYT -O 'encoder-4.pkl'
```

## Generating descriptions for images

You can find an example of using this caption generator at [Inference.ipynb](https://github.com/alex-f1tor/Image-Caption/blob/master/Inference.ipynb) notebook.

Few examples of generated captions for images:

![Image](https://github.com/alex-f1tor/Image-Caption/blob/master/imgs/bird_sample.png)

![Image](https://github.com/alex-f1tor/Image-Caption/blob/master/imgs/pizza_sample.png)


You can also:
  - Train your own caption network with MS-COCO dataset based on pipeline at [Training.ipynb](https://github.com/alex-f1tor/Image-Caption/blob/master/Training.ipynb)
  - Estimate model performance at [cocoEvalCap.ipynb](https://github.com/alex-f1tor/Image-Caption/blob/master/cocoEvalCap.ipynb) via different [metrics](https://github.com/tylin/coco-caption), like CIDEr, Rouge-L and etc.


## Model performance

The general estimation of captions quality generated for MS-COCO validation set by [CIDEr](https://arxiv.org/abs/1411.5726) metric:
![Image](https://github.com/alex-f1tor/Image-Caption/blob/master/imgs/CIDER_coco_valset.png)

