# Deploy Deeplearning Model with Flask-Restplus, Swagger as Demo app
[![Apache license](https://img.shields.io/badge/license-Apache-blue)](http://perso.crans.org/besson/LICENSE.html)
[![](https://img.shields.io/badge/python-3.6%2B-green.svg)]()

> A BeautyGAN and MobileNet model using web demo based flask, flask-restplus, swagger

## Introduction
------------------
*BeautyGAN*: Instance-level Facial Makeup Transfer with Deep Generative Adversarial Network

- Website: [http://liusi-group.com/projects/BeautyGAN](http://liusi-group.com/projects/BeautyGAN)

*MobileNetV2*: Computer Vision and PatternRecognition

- Website: [https://arxiv.org/abs/1801.04381](https://arxiv.org/abs/1801.04381)

Usage

- Python 3.6+
- Tensorflow 1.x

Download pretrained models

- [https://pan.baidu.com/s/1wngvgT0qzcKJ5LfLMO7m8A](https://pan.baidu.com/s/1wngvgT0qzcKJ5LfLMO7m8A)
- [https://drive.google.com/drive/folders/1pgVqnF2-rnOxcUQ3SO4JwHUFTdiSe5t9](https://drive.google.com/drive/folders/1pgVqnF2-rnOxcUQ3SO4JwHUFTdiSe5t9)

Save pretrained model, index, checkpoint to `models/model_p2`

```
.
+--docs
+--mlib
+--models
|   +-- model_dlib
|   +-- model_p2 -> make folder, here save models
+--static
...
```
## Screenshots
------------------
Swagger main

![intro1](docs/intro1.png "intro1")

BeautyGAN main

![intro2](docs/intro2.png "intro2")


## Getting Started
------------------

#### 1.Download or clone source code from github.
```shell
$ cd flask-deeplearning-service-demo
```
#### 2.Install Python packages
```shell
$ pip install -r requirements.txt
```
#### 3. Run(flask nomarl mode or flask-restapi mode)
```shell
$ python app.py
or
$ python app_rest.py
```
It will deploy :

- Flask app running in [http://localhost:8080](http://localhost:8080)
- Flask normal mode(optional)
- Flask rest-plus mode with swagger(optional)

## Features
------------------
- Responsible UI
- Support image drag-and-drop
- Use vanilla JavaScript, HTML and CSS
- RestAPI exception handling
- Dlib facing shape predictor landmark

## Demos
------------------
Image classification demo

![demo1](docs/demo1.gif "demo1")

Image beautyGAN demo

![demo2](docs/demo2.gif "demo2")

## Dependencies:
------------------
- [BeautyGAN](http://liusi-group.com/projects/BeautyGAN)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)
- [dlib](http://dlib.net/)  
- [opencv-python](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html)

## License
------------------
[GPL License](LICENSE)