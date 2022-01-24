# Variational AutoEncoder(VAE) implemented with tensorflow 2

## Description
Auto-Encoding Variational Bayes[[link](https://arxiv.org/abs/1312.6114)]

## structure

```
.
└── img/
    └──batch/
        ├── image_at_batch_{epoch}_{batch}.png   # Generated image at {epoch} {batch}
        └── ...
    ├── image_at_epoch_{num}.png                 # Generated image at {epoch}
    ├── ...
    └── vae.gif                                  # Animation of generated images
└── utils/           
    ├── __init__.py
    ├── callback.py                              # Custom callback function
    ├── layers.py                                # Encoder, Decoder implement
    ├── model.py                                 # VAE implement
    └── utils.py                                 # Other functions to draw a plot
├── .gitignore         
├── requirements.txt   
├── config.txt                                   # model config
├── VAE.ipynb                                    # Examples of progress 
└── VAE.py                                       # model training and save weight py
```

## Usage

```
python VAE.py --fig_save=True --model_save=True
```

+ --fig_save : Whether to save the generated image(bool, default=False)  
+ --model_save : Whether to save the generated model(bool, default=True)  

## Result

+ Generated image

![img](./img/vae.gif)

## reference
Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).