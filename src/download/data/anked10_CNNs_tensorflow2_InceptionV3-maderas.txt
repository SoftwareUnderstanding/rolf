# anked10-CNNs_tensorflow2_InceptionV3-maderas

Redes neuronales Convolucionales utilizadas para el reconocimiento de Especies maderables Mediante Imágenes

Algoritmos **InceptionV3** implementado usando  TensorFlow-2.0

##Entrenamiento

1.  Requerimientos
+ Python >= 3.6
+ Tensorflow == 2.0.0
  
2.  Para entrenar a ResNet en su propio conjunto de datos, puede colocar el conjunto de datos en la carpeta **original_dataset**, y el directorio debería verse así

  ```
|——original dataset
   |——class_name_0
   |——class_name_1
   |——class_name_2
   |——class_name_3
```
3.  Ejecute el script **split_dataset.py** para dividir el conjunto de datos sin procesar en conjunto de datos de entrenamiento (train), conjunto de datos de validacion(valid) y conjunto de datos de prueba (test).

```
|——dataset
   |——train
        |——class_name_1
        |——class_name_2
        ......
        |——class_name_n
   |——valid
        |——class_name_1
        |——class_name_2
        ......
        |——class_name_n
   |—-test
        |——class_name_1
        |——class_name_2
        ......
        |——class_name_n
```
4.  Modifique los parametros como corresponda en  **config.py**.
5.  Ejecute el script **train.py** para empezar el entrenamiento.


## Evaluación de algoritmos
Ejecute **evaluate.py** para evaluar el rendimiento del modelo en el conjunto de datos de prueba. 




## Referencias
1. Paper original :https://arxiv.org/abs/1512.00567
2. Google official implementation of InceptionV3 (TensorFlow 1.x): https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v3.py
3. https://www.jianshu.com/p/3bbf0675cfce
4. Official PyTorch implementation of InceptionV3 : https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
