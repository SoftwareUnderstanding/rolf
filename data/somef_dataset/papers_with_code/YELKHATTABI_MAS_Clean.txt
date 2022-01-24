# MAS_Clean

This report implement a very simple API to make model predictions 
The content so far let you make prediction on images using 
VGG_16 which is an image classification model
SSD300 which is an object detector



## Content of the repo

```
├── README.md
├── VGG_16_cat_and_dogs
│   ├── VGG_16_predictor.py 
│   └── model_weights ==> Weights to be downloaded
├── flask_apps
│   ├── __pycache__
│   ├── predict_app_ssd.py ==> to launch inorder to set the server for object detection 
│   ├── predict_app_vgg.py ==> to launch inorder to set the server for image classification
│   └── static
├── requirements.txt ==> Requirement to be install
└── ssd_keras ==> Library to be installed
    ├── model_weights ==> Weights to be downloaded
    ├── setup.py
    ├── ssd_keras
    ├── ssd_keras.egg-info
    ├── ssd_keras_readme.md
    └── ssd_predict.py

```

## Setting the environment 
Using your favorite virtual environment manage, set a new environment and start by installing the requirement by running the following command 
```
pip install -r requirement.txt
```
This command will automatically install the library `ssd_keras`
You can verify that the library is installed by using the command
```
$ pip list | grep ssd_keras
```
If the library is not installed, you can installed manually launching the following command being inside the repo

```
pip install -e ./ssd_keras/.
``` 
This will install the library locally without dublicating the files of the library

once the requirement are installed you can run the following command to make sure that all needed libraries are installed
## Download pretrained weights
Download VGG16 weights (`.h5` file) into the folder `./VGG_16_cat_and_dogs/model_weights/.` from the following link 
https://drive.google.com/uc?id=19yICdtSbU_YkQBRxJ2if9KJwUL1oY5xs&export=download

Download SSD300 weights (`.h5` file) into the folder `./ssd_keras/model_weights/.` from the following link 
https://drive.google.com/uc?id=121-kCXaOHOkJE_Kf5lKcJvC_5q1fYb_q

## Make simple prediction

### Image classificaiton with pretrained VGG 16
Paper of VGG16 https://arxiv.org/pdf/1409.1556.pdf
Simplified explanation of the model architecture https://neurohive.io/en/popular-networks/vgg16/

To run a simple prediction, use the script `./VGG_16_cat_and_dogs/VGG_16_predictor.py` 
You can simply run the script using `-w` argument for weights path, and -i `-i` for input image path

### Object Detection with pretrained SSD300
Model paper https://arxiv.org/pdf/1512.02325.pdf
Interesing blog about single shot detectors models family https://towardsdatascience.com/review-ssd-single-shot-detector-object-detection-851a94607d11

To run a simple prediction, use the script `./ssd_keras/ssd_predict.py` 
You can simply run the script using `-w` argument for weights path, and -i `-i` for input image path
the output will be the original image with model predictions annotations

## Make prediction through the web-app

### Image classification
Run the script `./flask_apps/predict_app_vgg.py` then oppen the link http://0.0.0.0:5000/static/predict_classification.html 
The webpage will enable you to load your beloved image and get model predictions

### Object detection 
Run the script `./flask_apps/predict_app_ssd.py` then oppen the link http://0.0.0.0:5001/static/predict_object_detection.html
The webpage will enable you to load your beloved image and get model predictions

references : 
https://github.com/pierluigiferrari/ssd_keras
