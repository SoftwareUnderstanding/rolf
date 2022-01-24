# EfficientDet: Scalable and Efficient Object Detection
Implementación en PyTorch del algoritmo EfficientDet para detección de objetos. En particular se entreno el modelo para el reconocimiento de tres clases (Stationery):
* Pens and Pencils.
* Scissors.
* Notebook.

**Autor:**  M. en C. Mario Xavier Canche Uc, Septiembre 2020, *mario.canche@cimat.mx*  
**Basado en:** https://arxiv.org/pdf/1911.09070.pdf 

**Material de referencia:**
- https://towardsdatascience.com/a-thorough-breakdown-of-efficientdet-for-object-detection-dc6a15788b73
- https://blog.roboflow.com/training-efficientdet-object-detection-model-with-a-custom-dataset/
- https://colab.research.google.com/drive/1ZmbeTro4SqT7h_TfW63MLdqbrCUk_1br#scrollTo=KwDS9qqBbMQa

## ¿Cómo funciona?
En primer lugar, se propone una red piramidal de características bidireccionales ponderadas (BiFPN), que permite una fusión de características de múltiples escalas fácil y rápida; En segundo lugar, proponen un método de escalamiento compuesto que escala uniformemente la resolución, la profundidad y el ancho de todas las redes troncales, redes de entidades y redes de predicción de cajas / clases al mismo tiempo.  
El EfficientDet logra un nuevo AP de COCO al 55,1% de última generación con muchos menos parámetros y FLOP que los detectores anteriores.

<img src="images/flops.png" width="500">

Código del paper disponible en: https://github.com/google/automl/tree/master/efficientdet.

## Arquitectura del EfficientDet
<img src="images/network.png" width="800">

## Resultados
Se realizó el entrenamiento con imágenes de objetos de papelería (tijeras, lápices y libretas) sobre una mesa.
La complejidad del reconocimiento de estos objetos esta en la similitud de algunos de ellos y la sobreposición de todos ellos en un espacio pequeño.  
<img src="images/output0.jpg" width="300">
<img src="images/output1.jpg" width="300">
<img src="images/output2.jpg" width="300">

