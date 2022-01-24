# BREIN Deep Leaning Challenge

By [Arian Gallardo](http://github.com/ariangc).

Pontificia Universidad Catolica del Peru (PUCP).

### Tabla de contenidos
0. [Introduccion](#introduccion)
0. [Problema presentado](#problema-presentado)
0. [Solucion](#solucion)
0. [Resultados](#resultados)
0. [API](#api)

### Introduccion

Este repositorio contiene los archivos que corresponden a la solucion del reto presentado para el proceso de seleccion del Hub de Innovacion del Grupo Breca (BREIN), para Febrero 2020. 

### Problema presentado

En este reto se busca ayudar a una empresa retail a mejorar su proceso de manejo de inventarios. La empresa esta buscando una manera de reducir el esfuerzo humano en la clasificacion de sus productos.

A traves de un clasificador de imagenes, podemos ayudar a la compania a analizar su inventario.

Todos los datos se pueden encontrar en el siguiente enlace: [Data Reto Brein](https://www.dropbox.com/s/kub6cebbsgiotla/reto_deep_learning.rar?dl=0)

Se presentan imagenes de productos que corresponden a 25 clases (agua, arroz, aceite, etc).

### Solucion

Se uso el modelo ResNet-18 descrito en el paper "Deep Residual Learning for Image Recognition" (http://arxiv.org/abs/1512.03385). Este modelo, asi como la familia de ResNets, fueron usados en [ILSVRC](http://image-net.org/challenges/LSVRC/2015/) y [COCO](http://mscoco.org/dataset/#detections-challenge2015), competencias de Computer Vision en 2015, en las que ganaron el 1er puesto en: Clasificacion de ImageNet, Deteccion de ImageNet, Localizacion de Imagenet, Deteccion en COCO, y Segmentacion en COCO.

Se uso una distribucion de 63%-27%-10% del dataset para train, validation y test respectivamente. El preprocesamiento de imagenes consistio en resizing, normalization y data augmentation.  

### Resultados

0. Curvas de accuracy sobre el dataset (20 epocas, learning rate = 0.00005, optimizador = Adam)
	![Training acc curves](https://raw.githubusercontent.com/ariangc/breinchallenge/master/models/pytorch_resnet18/train_val_acc.jpg)

0. Curvas de loss sobre el dataset (20 epocas, learning rate = 0.00005, optimizador = Adam)
	![Training loss curves](https://raw.githubusercontent.com/ariangc/breinchallenge/master/models/pytorch_resnet18/train_val_loss.jpg)

0. Matriz de confusion no normalizada (test = 10% del dataset)
	![Confusion matrix 1](https://raw.githubusercontent.com/ariangc/breinchallenge/master/models/pytorch_resnet18/cm.jpg)

0. Matriz de confusion normalizada (test = 10% del dataset)
	![Confusion matrix 2](https://raw.githubusercontent.com/ariangc/breinchallenge/master/models/pytorch_resnet18/cm_normalized.jpg)

### API

Para usar este modelo para clasificar imagenes contenidas en un directorio, instalar las dependencias en la carpeta **client/requirements.txt** con el comando **pip install -r path-to-requirements.txt**, y luego ejecutar el siguiente comando: `python main.py --data_path DATA_PATH` donde **DATA\_PATH** es el directorio conteniendo las imagenes. El resultado se guardara en el archivo **results.csv**.

