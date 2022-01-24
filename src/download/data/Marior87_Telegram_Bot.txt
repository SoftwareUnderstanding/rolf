# ML usando un Bot de Telegram

Este repo contiene el desarrollo de un bot para telegram que cuenta con dos funcionalidades: Clasificación de imágenes y Transferencia de Estilo. Utilizaremos la siguiente imagen como ejemplo:

Imagen Ejemplo:
<p align="center">
<img src="assets/ejemplo.jpg" width=500px height=350px/>
</p>
Los comandos para interactuar con el bot son:

1. /clas
2. /style


### /clas: 

Inicia una conversación en donde se solicita una imagen cualquiera y el bot devuelve una predicción del objeto que aparezca en la imagen y un 'mapa de atención' donde señala en un heatmap las zonas de la imagen más relevantes para la decisión.

Aquí utilizamos un modelo preentrenado sobre ImageNet de MobileNetV2 (https://arxiv.org/abs/1801.04381) para generar la predicción, y una técnica de representación de conceptos visuales para CNNs llamada 'mapas de atención'. (https://arxiv.org/pdf/1612.03928.pdf)

Para mapas de atención, sugiero revisar el notebook de Colab:
https://colab.research.google.com/github/zaidalyafeai/AttentioNN/blob/master/Attention_Maps.ipynb#scrollTo=UPDeShtiuYsz

Resultado:<br><br>
    Predicción: "goldfish, Carassius auratus"<br>


Mapa de Atención:<br>
<p align="center">
<img src="assets/ejemplo_mapa_atencion.jpg" width=500px height=350px/>
</p>

### /style: 

Inicia una conversación en donde se solicita una imagen y el bot devuelve la misma imagen estilizada tomando el estilo que se extrae del archivo 'estilo1.jpg' (Se puede utilizar otro archivo para obtener resultados diferentes).

Imagen para Estilo:
<p align="center">
<img src="estilo1.jpg" width=500px height=350px/>
</p>
En este caso se utiliza un modelo preentrenado del TensorFlow Hub:

https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1

En la documentación de TensorFlow se muestra un tutorial para esto:

https://www.tensorflow.org/tutorials/generative/style_transfer

Resultado:
<p align="center">
<img src="assets/ejemplo_estilizado.jpg" width=500px height=350px/>
</p>

## Configuración para uso:

* Crear una carpeta de trabajo.
* Crear un entorno virtual de Python (versión 3.6) y activarlo.
* Instalar los paquetes indicados en requirements.txt
* Clonar este repositorio.
* Obtener un token para un bot de telegram (yo usé al BotFather).
* Crear la variable de entorno 'TGM_BOT_TEST' y darle como valor el token obtenido (puedes cambiar el nombre de la variable de entorno a utilizar en the_bot.py)
* Ejecutar en la consola:
```
python the_bot.py
```

Una vez realizados estos pasos (y si no hay errores), inicia una conversación en telegram con tu bot y empieza a interactuar utilizando los comandos /clas y /style en el chat.

**Nota:** Puesto que es un proyecto cuya intención es demostrar la funcionalidad de las herramientas, fue creado con sentido 'local'. Por ello, los archivos de imagen (entre otras cosas) son guardados con nombres genéricos, lo cual no es funcional si el bot quiere ser utilizado en un grupo.
