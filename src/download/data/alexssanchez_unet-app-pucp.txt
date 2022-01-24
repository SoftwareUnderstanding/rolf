# unet-app-pucp

Revisión y aplicación de una red U-Net y su evolción a U-Net++

### [U-Net: Convolutional Networks for Biomedical Image Segmentation (2015)](https://arxiv.org/abs/1505.04597v1)

La U-Net es una de las famosas Fully Convolutional Networks (FCN) en segmentación de imágenes biomédicas, la cual fue publicada en 2015. 
La anotación de imágenes biomédicas, siempre se necesita expertos que hayan adquirido los conocimientos relacionados, considerando que es un proceso manual y toma mucho tiempo.
Si el proceso de anotación de imágenes se realiza de manera automática, se logra un menor esfuerzo y se reduce el error por parte de los expertos, también se estaría reduciendo los costos.

Diagrama de Aquitecura de una red U-Net:

![](image/Arq_UNet.png)

Para citar a esta arquitectura de red, utilice: 
```
@misc{ronneberger2015unet,
    title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
    author={Olaf Ronneberger and Philipp Fischer and Thomas Brox},
    year={2015},
    eprint={1505.04597},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
### [UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation (2020)](https://arxiv.org/abs/1912.05074v2)

Evolución de U-Net a UNet ++. Cada nodo en el gráfico representa un bloque de convolución, la dirección de las flechas indican en qué dirección se realiza el el muestreo y las flechas de puntos indican conexiones salteadas.

Diagrama de Aquitecura de una red U-Net++, utilice:

![](image/Arq_UNet++.png)

Para citar a esta arquitectura de red: 
```
@misc{zhou2019unet,
    title={UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation},
    author={Zongwei Zhou and Md Mahfuzur Rahman Siddiquee and Nima Tajbakhsh and Jianming Liang},
    year={2019},
    eprint={1912.05074},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
```

### Descripción de Dataset

Para la presente aplicacón se está considerando el dataset de Hipocampo, el cual fue publicado en la [Medical Segmentation Decathlon](http://medicaldecathlon.com/), para mayor detalle revisar el paper publicado en [ArXiv](https://arxiv.org/abs/1902.09063v1).

El conjunto de datos está compuesto de Imágenes por Resonancia Magnética (IRM), el cual se subdivide de la siguiente manera:

Descripción       | # Imágenes
:----------------:|:-----------:
Adultos Sanos     |  90
Adulto Enfermos   |  105
Total Imágenes    |  205

Las Imágenes de los Adultos enfermos se subdividen según la enfermedad que sufren, los cuales se muestran en la siguiente manera:

Descripción                                      | # Imágenes
:-----------------------------------------------:|:-----------:
Trastorno psicótico no afectivo (esquizofrenia)  |  90
Trastorno esquizoafectivo                        |  105
Trastorno esquizofreniforme                      |  205
Total Imágenes                                   |  205 

Todas las personas consideradas estaban libres de enfermedades médicas o neurológicas significativas, lesiones en la cabeza y/o uso dependencia de sustancias activas. 

Características de las imágenes:

Descripción                   | Valor             | Observación
:----------------------------:|:-----------------:|:---------------------------------------------:
Tipo de Secuencia             |  MPRAGE           | ponderada en 3D T1 (TI / TR / TE, 860 / 8.0 / 3.7 ms; 170 cortes sagitales; tamaño de vóxel, 1.0 mm3)
Tipo de Escáner               |  Philips Achieva  | Philips Healthcare, Inc

Donde: 
- TI: Medium Inversion Time
- TR: Long Repetition Time
- TE: Rapidly Acquired Gradient Echoes

### Segmentación de Imágenes utilizando [Image Segmentation Keras](https://github.com/divamgupta/image-segmentation-keras)

[Image Segmentation Keras](https://github.com/divamgupta/image-segmentation-keras) es un paquete escrito en [pyton](https://www.python.org/) que implementa Segnet, FCN, UNet, PSPNet, entre otros modelos de segmentación de imágenes [Keras](https://keras.io/).

El siguiente ejemplo de segmentación de imágenes toma como guia el post [A Beginner's guide to Deep Learning based Semantic Segmentation using Keras](https://divamgupta.com/image-segmentation/2019/06/06/deep-learning-semantic-segmentation-keras.html).

Imágen Original                                  | # Imágen Segmentada
:---------------------------------------:|:---------------------------------------:
![](image/moto_gp_01.jpg)                |![](image/out.png)

Para realizar la segmentación de la imagen mostrada, se utilizó la red pre-entrenada ["pspnet_101_voc12"](https://github.com/divamgupta/image-segmentation-keras/blob/master/keras_segmentation/pretrained.py#L67) el cual usa como el dataset a [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/).

