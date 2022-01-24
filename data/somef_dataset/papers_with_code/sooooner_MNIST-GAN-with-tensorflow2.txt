# Generative Adversarial Networks(GAN) implemented with tensorflow 2

## Description
Generative Adversarial Networks[[link](https://arxiv.org/abs/1406.2661)]

## structure

```
.
└── img/
    ├── image_at_epoch_{num}.png                 # Generated image at {epoch}
    ├── ...
    └── gan.gif                                  # Animation of generated images
└── utils/           
    ├── __init__.py
    ├── callback.py                              # Custom callback function
    ├── layers.py                                # Encoder, Decoder implement
    ├── model.py                                 # VAE implement
    └── utils.py                                 # Other functions to draw a plot
├── .gitignore         
├── requirements.txt   
├── config.py                                   # model config
├── GAN.ipynb                                    # Examples of progress 
└── GAN.py                                       # model training and save weight py
```

## Usage

```
python GAN.py --model_save=True
```
 
+ --model_save : Whether to save the generated model weight(bool, default=True)  

## Result

+ Generated image

![img](./img/gan.gif)

## reference
Goodfellow, Ian J., et al. "Generative adversarial networks." arXiv preprint arXiv:1406.2661 (2014).