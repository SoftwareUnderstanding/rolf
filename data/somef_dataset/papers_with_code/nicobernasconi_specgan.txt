# SpecGAN - generate audio with adversarial training



Preprocesar Audio
---

Se generan los archivos de entrenamiento,generando los diagramas mels de los audios en las carpeta de entrenamiento, codificando ademas las clases de dichos audio.
Se debe orgaanizar los audios en una estructura de carpertas con la forma: {TRAIN_AUDIO_FOLDER}/{CLASSES}/{AUDIO_FILE}
El preproceso dara por resultado, un archivo con extencion .npz

```
$ python preprocess_audio.py -i ./drums-selected/ -o ./training_data.npz
```


Entrenamiento
---

Se inicia el entrenamiento    

```
$ python specgan_categorical.py -i ./training_data.npz -o ./checkpoints -e 200
```

```
-i Archivo creado en el proceso iniciado en  proprocess_audio.py
-e numero de epecas
-o Carpeta en la que se generaran audios, espectogramas y modelos, productos del entrenamientos
```



Creditos 🙏
---

- The implementation of Improved Wasserstein GAN proposed in https://arxiv.org/abs/1704.00028 was based on [a keras implementation](
https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py).

- Spectrogram to Audio conversion (Griffin-Lim algorithm) was based on [Yunchao He's implementation](https://github.com/candlewill/Griffin_lim).
