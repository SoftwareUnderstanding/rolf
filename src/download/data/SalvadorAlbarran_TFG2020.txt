# TFG2020
## Aceleración de AI en dispositivos de bajo consumo


En este repositorio se puede encontrar tanto los modelos usados para este trabajo, como los código para la ejecución de la aplicación y el firmware utilizado en las placas Maix Go y Maix Bit.

A continuación se detalla la instalación, entrenamiento y ejecución del proyecto:



# ENTRENAR, CONVERTIR Y EJECUTAR MODELO MOBILENET
Tutorial sacado de: https://en.bbs.sipeed.com/t/topic/682/

Autor: Salvador Albarrán Tiradas

Sistema operativo: Ubuntu 18.04

Para Maix Go y Maix Bit:

TENSORFLOW Y KERAS

https://www.tensorflow.org/

https://keras.io/
 
**Tensorflow** es una plataforma de código abierto para el aprendizaje automático que cuenta con un ecosistema integral y flexible de herramientas, bibliotecas y recursos comunitarios para impulsar el estado del arte y desarrollar fácilmente aplicaciones basadas en aprendizaje automático.

**Keras** es una API de redes neuronales de alto nivel, escrita en Python y capaz de ejecutarse sobre TensorFlow, CNTK o Theano.
Admite redes convolucionales y redes recurrentes, así como combinaciones de las dos y se ejecuta sin problemas en CPU y GPU.

# 1. Instalar el entorno (Se ha elegido Keras)
Nota: La versión de tensorflow 1.x a 2.x cambia la forma de escribir los scripts, por lo que si se usan los scripts que se dan en este ejemplo con la versión 2.x no funcionará, los cambios son básicamente que ya no hay que instalar Keras ya que está implementado dentro de tensorflow, por lo que para usar una librería de keras tendríamos que hacer:

Import tensorflow as tf

tf.keras.optimizers…

más información aquí: https://www.tensorflow.org/guide/migrate/

Tenemos varias opciones:

**Docker tensorflow (Recomendada por Sipeed):**

Aquí recomiendan la versión de tensorflow 1.13.1, para usarlo se necesita actualizar los drivers de la gráfica, aquí tenéis la guía de instalación de tensorflow:
https://www.tensorflow.org/install/docker
 
**Instalación mediante pip:**

Se necesita instalar tensorflow-gpu == 1.13.1 si quieres usar la misma versión que recomiendan en sipeed con la versión de gpu, si quieres usar la versión de cpu hay que eliminar la parte de gpu: tensorflow==1.13.1.
Además, necesitaremos instalar el software de NVIDIA donde se puede encontrar la información aquí: https://www.tensorflow.org/install/gpu/
Para saber que versión instalar se puede mirar en esta gráfica:

![](../master/imagenes/Image.png)

Sacada de https://www.tensorflow.org/install/source#linux


**Instalación mediante conda (Anaconda, la que he usado yo):**

Parecido a la instalación mediante pip pero con más facilidades, para instalar Tensorflow gpu basta con poner el siguiente comando:

conda install tensorflow-gpu==1.13.1

Con esto instalará todos los requisitos de NVIDIA que necesita esta versión de Tensorflow, como en pip se necesitará que tengas actualizados los drives de tu gráfica, pero no tendrás que instalar uno a uno todos los componentes.

Además, en el caso de que se use la versión de CPU (intel) anaconda tiene un rendimiento bastante significativo (alrededor de 8 veces más rápido) en comparación con pip como se muestra en la siguiente figura:

![](../master/imagenes/Image%20%5B1%5D.png)

Este ejemplo se realizó con un Intel® Xeon® Gold 6130.

La información se puede encontrar aquí:

https://www.anaconda.com/tensorflow-in-anaconda/



# 2. Descargar el modelo pre entrenado de mobilenetv1

https://arxiv.org/pdf/1704.04861.pdf

https://ai.googleblog.com/2017/06/mobilenets-open-source-models-for.html
 
Mobilenetv1 son clases de modelos eficientes llamados MobileNets para aplicaciones de móviles y aplicaciones de visión empotradas. Se basa en una arquitectura optimizada que utiliza convoluciones separables en profundidad para construir redes profundas de peso ligero.

Introducen 2 hiperparámetros:

Multiplicador de anchura: Se usa para adelgazar una red de manera uniforme en cada capa. Dada una capa y un multiplicador de anchura α el número de canales de entrada M sería αM y el número de canales de salida N sería αN.
Esto tiene el efecto de reducir el coste computacional y el número de parámetros cuadráticamente alrededor de α2. Se puede aplicar a cualquier estructura de modelo para definir un nuevo modelo más pequeño.

Multiplicador de resolución (ρ): Se aplica en la imagen de entrada y la interna representación de cada capa es reducida por dicho multiplicador.
Aplicamos este hiperparámetro implícitamente cuando ajustamos la resolución de entrada, es decir al poner como resolución 224,192,160 o 128.
Esto hace reducir el coste computacional alrededor de ρ2.

La arquitectura de mobilenet se puede observar en la siguiente tabla:

![](../master/imagenes/Image%20%5B2%5D.png)

https://github.com/fchollet/deep-learning-models/releases/tag/v0.6/

Se recomienda usar mobilenet_7_5_224_tf_no_top.h5

Lo guardaremos en ~ /.keras/models/

Significa que tiene un valor alpha = 0.75 llamado multiplicador de anchura, el tamaño de la imagen de entrada es de 224x224 y sin capa del top(sin dropout ni última capa conectada con activación softmax).

Se recomienda por la siguiente tabla:

![](../master/imagenes/Image%20%5B3%5D.png)


La diferencia entre un alpha 1.00x  a 0.75x es de más o menos un 2% de pérdida de precisión, mientras que el número de parámetros se reduce de 4.24M a 2.59M, que usando una cuantización de 8 bits se resume en 4.25MB vs 2.59MB, una diferencia considerable.

Usando Micropython no podríamos usar el de 4.35MB, pero sí el de 2.59MB, en cambio sí se usa Standalone o FreeRTOS se podría usar ambos sin problema.

Yo he probado a entrenar ambos modelos, que los comentaré más adelante.


# 3. Descargar el conjunto de datos imagenet ILSVRC2012_IMG_TRAIN

Son 150GB así que deja espacio suficiente en disco.

Este conjunto de datos contiene 1000 clases, al descomprimirlo tendrás 1000 .tar por lo que será necesario usar un script para agilizar la operación:

![](../master/imagenes/Image%20%5B4%5D.png)


# 4. Ajustar el archivo original mobilenet.py
El chip k210 usa un relleno de ceros en todas las direcciones (arriba, abajo, derecha, izquierda) mientras que el relleno de Keras es solo de derecha y abajo, por lo que vamos a necesitar cambiar dos líneas al fichero original.

Se encuentra en ~ /python3.5/site-packages/keras_applications/mobilenet.py

En la función original tenemos:
x = layers.ZeroPadding2D(padding = ( ( 0, 1 ), ( 0, 1 ) ), name = ‘conv_pad_%d’ % block_id ) (inputs)

 Y nosotros queremos:
x = layers.ZeroPadding2D(padding = ( ( 1, 1 ), ( 1, 1 ) ), name = ‘conv_pad_%d’ % block_id ) (inputs)

Aparece 2 veces, en la línea 354 y en la 423.

# 5. Terminar script de entrenamiento
Como hemos escogido un modelo sin el top (sin softmax ni dropout) tenemos que añadirlo, sipeed nos proporciona un script hecho, aunque realmente lo que hace es:

+ Añadir el dropout y una capa final con activación softmax.

![](../master/imagenes/Image%20%5B5%5D.png)

Si queremos usar el modelo con 1.00x de channel count tenemos que cambiar Alpha a ese mismo valor.
 
+ Arreglar los pesos de las capas anteriores.

![](../master/imagenes/Image%20%5B6%5D.png)

+ Guardar el modelo

![](../master/imagenes/Image%20%5B7%5D.png)


En mi caso, tengo una Nvidia 1060 6GB por lo que la línea donde se paraleliza el modelo no la uso, así que si tienes múltiples gráficas te servirá para acelerar el proceso, en mi caso no, por lo que basta con eliminar la parte de: paralleled_ quedando, por ejemplo:

            model.summary()
            
El script se puede encontrar en su github:

https://github.com/sipeed/Maix-Keras-workspace/tree/master/mbnet

archivo mbnet_keras.py

Algunos parámetros que he modificado en el entrenamiento:

batch_size: Número de ejemplos de entrenamiento utilizados en una iteración.

epotch: Es un paso completo a través de los datos de entrenamiento.

step_size_train: Número de pasos dentro de un epotch.

Ejecutamos mbnet_keras.py y empezará a entrenar.

EJEMPLOS

He realizado 3 entrenamientos,

**1º** Sin modificar ningún parámetro y usando el modelo con 0.75x , donde he conseguido una precisión de 0.60 tardando algo más de 4 horas.
Resultado esperado ya que era lo previsto del ejemplo por el cual partíamos como referencia.

**2º** Usando el modelo es el de 1.00x  donde conseguí una precisión de 0.62 tardando 3 horas 24 min.
Un resultado esperado ya que en la tabla podíamos apreciar que aumentábamos mucho el número de parámetros, pero tendríamos una diferencia de un 2% aproximadamente y es lo que hemos obtenido.

![](../master/imagenes/Image%20%5B8%5D.png)


En este modelo, he necesitado cambiar el batch size de 512 a 256, ya que por la cantidad de memoria de mi gpu no podía distribuir tantos datos, significando un error de OOM (Out Of Memory) y no podía entrenarlo con mi gráfica, bajando este parámetro funcionó sin problemas.

**3º** Usando el modelo de 0.75x cambiando el batch size y el número de epotch. En este caso decidí aumentar por 4 este número, siendo así epotch = 80 y un batch size de 256 intentando conseguir una mejora, pero solo conseguí un 0.63 de precisión (vs el 0.60 del primer ejemplo) tardando alrededor de 8 horas en entrenarlo.

Intentando ver si aumentando el parámetro epotch observamos una mejora de precisión:

![](../master/imagenes/Image%20%5B9%5D.png)

# 6. Convertir a kmodel

Las herramientas las proporciona sipeed, aquí podéis encontralas:

https://github.com/sipeed/Maix_Toolbox/

Maix Toolbox es una colección de herramientas para modelos (scripts).

Se puede usar para convertir el modelo de un formato a otro, lo podemos ver en la siguiente tabla:

![](../master/imagenes/Image%20%5B10%5D.png)


Primero obtenemos las herramientas con el script get_nncase.sh

NNcase es un compilador de modelos y herramienta de inferencia.

Tiene 2 comandos ‘compile e ‘infer’

‘compile’ compila el modelo entrenado (.tflite, .caffemodel) a kmodel.

Kmodel tiene 2 versiones: kmodelV3 y kmodelV4.

KmodelV3 es la generada por la versión anterior de nncase a 0.2 y es la versión más optimizada, ya que la versión kmodelV4 a pesar de soportar más operaciones, más modelos cuesta más memoria (alrededor de 360KBde ram ), está en desarrollo luego hay operaciones que no están tan bien como el la V3 y tiene una latencia mayor (37ms vs  111ms)

La diferencia de fps entre las dos versiones es notoria (V3: ~24.05 fps V4: 8.45 fps).

Tiene los siguientes parámetros:

![](../master/imagenes/Image%20%5B11%5D.png)


<input file> dirección del modelo de entrada.

<output file> dirección del modelo de salida.
 
**-i, --input-format** opción usada para espeficicar el formato del modelo de entrada. Nncase soporta modelos tflite y caffe.

**-o, --output-format** opción usada para especificar el formato de salida del modelo, solo hay una opción: kmodel.

**-t, --target** La opción se utiliza para configurar el dispositivo de destino deseado para ejecutar el modelo. CPU es el objetivo más general que casi todas las plataformas deberían admitir. k210 es la plataforma Kendryte K210 SoC. Si se configura esta opción en k210, este modelo solo puede ejecutarse en K210 o ser emulado en el PC.

**--inference-type** se configura a float si se quiere precisión a coste de más memoria y pérdida de aceleración en el K210 KPU.Configurar a unit8 si se quiere mayor aceleración del KPU y mayor velocidad, se necesita dar un conjunto de datos para la cuantización de calibración para cuantizar los modelos más tarde.

**--dataset** es para proporcionar tu conjunto de datos de calibración de cuantización para cuantificar tus modelos. Debe poner cientos o miles de datos en el conjunto de entrenamiento en este directorio. Solo se necesita esta opción cuando se establece --inference-type en uint8.

**--dataset-format** es para establecer el formato del conjunto de datos de calibración. El valor predeterminado es ‘image’, nncase usará opencv para leer sus imágenes y escalar automáticamente al tamaño de entrada deseado de tu modelo. Si la entrada tiene 3 canales, ncc convertirá imágenes a float tensors RGB [0,1] en diseño NCHW. Si la entrada tiene solo 1 canal, ncc escalará en gris sus imágenes. Establezca en ‘raw’ si el conjunto de datos no es un conjunto de datos de imagen, por ejemplo, audio o matrices. En este escenario, debes convertir tu conjunto de datos a archivos binarios sin formato que contengan float tensors. Solo necesita esta opción cuando se establece --inference-type en uint8.

**--input-std and --input-mean** es para establecer el método de preproceso en tu conjunto de datos de calibración. Como se dijo anteriormente, ncc primero convertirá las imágenes en float tensors RGB [0,1] en el diseño NCHW, luego ncc normalizará sus imágenes usando la fórmula y = (x - mean) / std. Hay una tabla de argumentos de referencia:

![](../master/imagenes/Image%20%5B12%5D.png)


**--calibrate-method** El método consiste en establecer el método de calibración deseado, que se utiliza para seleccionar los rangos de activación óptimos. El valor predeterminado es no_clip, ya que ncc usará el rango completo de activaciones. Si se desea un mejor resultado de cuantización, puede usar l2 pero tomará más tiempo encontrar los rangos óptimos.

**--input-type** es para establecer el tipo de datos de entrada deseado cuando se hace la inferencia. El valor predeterminado es igual al tipo de inferencia. Si --input-type es uint8, por ejemplo, debe proporcionar RGB888 uint8 tensors cuando se haga la inferencia. Si --input-type es float, debe proporcionar float tensors RGB en su lugar.

**--max-allocator-solve-secs** es para limitar el tiempo máximo de resolución cuando se realiza la mejor búsqueda de asignación. Si se excede el tiempo de búsqueda, ncc recurrirá para usar el primer método de ajuste. El valor predeterminado es 60 segundos, establecido en 0 para deshabilitar la búsqueda.

**--dump-ir** es una opción de debug. Cuando está activado, ncc producirá algunos archivos de gráficos .dot en el directorio de trabajo. Puede usar Graphviz o Graphviz Online para ver estos archivos.

El comando infer puede ejecutar el kmodel, y se usa como propósito de depuración. ncc guardará los tensors de salida del modelo en archivos .bin en NCHW.

"NCHW" significa data cuyo diseño es (batch_size, channel, height, width)

**<input file>** es tu dirección donde se encuentra tu kmodel.

**<output path>** es el directorio donde se producirá la salida.
 
**--dataset** es el directorio del conjunto de prueba.

**--dataset-format**, **--input-std** and **--input-mean** tienen el mismo significado que en el comando compile.

Ojo que hay un error en el script hay que cambiar a las siguientes líneas:

tar -Jxf ncc-linux-x86_64.tar.xz

rm ncc-linux-x86_64.tar.xz

También puedes descargarlo manualmente y poner los contenidos en la carpeta creada ncc.

enlace aquí: https://github.com/kendryte/nncase/releases  la versión utilizada es  NNCase Converter v0.1.0 RC5

Una vez tenemos todo preparado ya podemos convertir nuestro modelo a .pb con el siguiente comando:

./keras_to_tensorflow.py --input_model workspace/mbnet75.h5  --output_model workspace/mbnet75.pb

(Podemos obtener un grafo de nuestro modelo si se desea con el siguiente script) 

./gen_pb_graph.py workspace/mbnet75.pb

pb significa ‘protobuf’, este contiene tanto la definición de  grafo como los pesos del modelo.

Para convertir nuestro modelo.pb a modelo.tflite existe un script que te guía a obtener el comando pb2tflite.sh, al final el comando quedaría de la siguiente forma:
Podemos usar toco o tflite_convert

toco --graph_def_file=workspace/mbnet75.pb --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE --output_file=workspace/mbnet75.tflite --inference_type=FLOAT --input_type=FLOAT --input_arrays=input_1 --output_arrays=dense_1/Softmax --input_shapes=1,224,224,3

# TensorFlow Lite Converter

Una vez que tenemos un modelo de TensorFlow capacitado, el convertidor de TensorFlow Lite aceptará ese modelo y generará un archivo FlatBuffer de TensorFlow Lite. El convertidor actualmente admite SavedModels, gráficos congelados (modelos generados a través de freeze_graph.py) y archivos de modelo tf.Keras. El archivo FlatBuffer de TensorFlow Lite se puede enviar a los dispositivos del cliente, generalmente dispositivos móviles, donde el intérprete de TensorFlow Lite los maneja en el dispositivo. Este flujo se representa en el siguiente diagrama.

![](../master/imagenes/Image%20%5B13%5D.png)


Comandos para usar toco o tflite_convert:

**--output_file.** Type: string. Especifica la ruta completa del archivo de salida.

**--graph_def_file.** Type: string. Especifica la ruta completa del archivo GraphDef de entrada congelado usando freeze_graph.py.

**--saved_model_dir.** Type: string. Especifica la ruta completa al directorio que contiene el modelo guardado.

**--keras_model_file.** Type: string. Especifica la ruta completa del archivo HDF5 que contiene el modelo tf.keras.

**--output_format.** Type: string. Default: TFLITE. Especifica el formato del archivo de salida. Valores permitidos:

+ TFLITE: TensorFlow Lite FlatBuffer format.

+ GRAPHVIZ_DOT: Formato GraphViz .dot que contiene una visualización del gráfico después de las transformaciones del gráfico.

Tener en cuenta que pasar GRAPHVIZ_DOT a --output_format conduce a la pérdida de transformaciones específicas de TFLite. Por lo tanto, la visualización resultante puede no reflejar el conjunto final de transformaciones gráficas. Para obtener una visualización final con todas las transformaciones de gráficos, usar --dump_graphviz_dir en su lugar.

**--input_arrays.** Type: lista de cadenas separadas por comas. Especifica la lista de nombres de tensores de activación de entrada.
**--output_arrays.** Type: lista de cadenas separadas por comas. Especifica la lista de nombres de tensores de activación de salida.
Los siguientes indicadores definen las propiedades de los tensores de entrada. Cada elemento en el indicador --input_arrays debe corresponder a cada elemento en los siguientes indicadores según el índice.
**--input_shapes.** Tipo: lista separada por dos puntos de listas enteras separadas por comas. Cada lista de enteros separados por comas da la forma de una de las matrices de entrada especificadas en la convención TensorFlow.

Ejemplo: --input_shapes = 1,60,80,3 para un modelo de visión típico significa un batch size de 1, una altura de imagen de entrada de 60, un ancho de imagen de entrada de 80 y una profundidad de imagen de entrada de 3 (que representan canales RGB )

Ejemplo: --input_arrays = foo, bar --input_shapes = 2,3: 4,5,6 significa que "foo" tiene una forma de [2, 3] y "bar" tiene una forma de [4, 5, 6].

**--std_dev_values, --mean_values.** Tipo: lista de floats separados por comas. Estos especifican los parámetros de (de-)cuantización de la matriz de entrada, cuando se cuantifica. Esto solo es necesario si inference_input_type es QUANTIZED_UINT8.

El significado de mean_values y std_dev_values es el siguiente: cada valor cuantificado en la matriz de entrada cuantificada se interpretará como un número real matemático (es decir, como un valor de activación de entrada) de acuerdo con la siguiente fórmula:
real_value = (quantized_input_value - mean_value) / std_dev_value.
Al realizar la inferencia flotante (--inference_type = FLOAT) en una entrada cuantificada, la entrada cuantificada se descartaría inmediatamente por el código de inferencia de acuerdo con la fórmula anterior, antes de proceder con la inferencia flotante.

Cuando se realiza una inferencia cuantificada (--inference_type = QUANTIZED_UINT8), el código de inferencia no realiza la descuantificación. Sin embargo, los parámetros de cuantificación de todas las matrices, incluidas las de las matrices de entrada especificadas por mean_value y std_dev_value, determinan los multiplicadores de punto fijo utilizados en el código de inferencia cuantificado. mean_value debe ser un número entero cuando se realiza una inferencia cuantificada.

**--inference_type.** Tipo: string. Predeterminado: float. Tipo de datos de todas las matrices de números reales en el archivo de salida, excepto las matrices de entrada (definidas por --inference_input_type). Debe ser {FLOAT, QUANTIZED_UINT8}.
Este indicador solo afecta a las matrices de números reales, incluidas las matrices flotantes y cuantizadas. Esto excluye todos los demás tipos de datos, incluidas las matrices enteras simples y las matrices de cadenas. Específicamente:
Si FLOAT, las matrices de números reales serán de tipo float en el archivo de salida. Si se cuantificaron en el archivo de entrada, se descuantifican.
Si QUANTIZED_UINT8, las matrices de números reales se cuantificarán como uint8 en el archivo de salida. Si fueron flotantes en el archivo de entrada, entonces se cuantizaron.
**--inference_input_type.** Tipo: string. Tipo de datos de una matriz de entrada de número real en el archivo de salida. Por defecto, --inference_type se usa como tipo de todas las matrices de entrada. Flag está destinado principalmente a generar un gráfico de punto flotante con una matriz de entrada cuantificada. Se agrega un operador descuantificado inmediatamente después de la matriz de entrada. Debe ser {FLOAT, QUANTIZED_UINT8}.
La bandera se usa típicamente para modelos de visión que toman un mapa de bits como entrada, pero requieren inferencia de punto flotante. Para tales modelos de imagen, la entrada uint8 se cuantifica y los parámetros de cuantificación utilizados para tales matrices de entrada son sus parámetros mean_value y std_dev_value.

**--default_ranges_min, --default_ranges_max.** Tipo: punto flotante. Valor predeterminado para los valores de rango (min, max) utilizados para todas las matrices sin un rango especificado. Permite al usuario proceder con la cuantificación de archivos de entrada no cuantificados o cuantificados incorrectamente. Estas banderas producen modelos con baja precisión. Están destinados a una fácil experimentación con la cuantización a través de la "cuantificación ficticia".

**--drop_control_dependency.** Tipo: booleano. Valor predeterminado: verdadero. Indica si se deben eliminar las dependencias de control de forma silenciosa. Esto se debe a que TensorFlow Lite no admite dependencias de control.

**--reorder_across_fake_quant.** Tipo: booleano. Valor predeterminado: falso. Indica si se debe reordenar los nodos FakeQuant en ubicaciones inesperadas. Se utiliza cuando la ubicación de los nodos FakeQuant impide las transformaciones gráficas necesarias para convertir el gráfico. Resultados en un gráfico que difiere del gráfico de entrenamiento cuantificado, lo que puede causar un comportamiento aritmético diferente.

**--allow_custom_ops.** Tipo: string. Valor predeterminado: falso. Indica si se permiten operaciones personalizadas. Cuando es falso, cualquier operación desconocida es un error. Cuando es verdadero, se crean operaciones personalizadas para cualquier operación que sea desconocida. El desarrollador deberá proporcionarlos al tiempo de ejecución de TensorFlow Lite con un solucionador personalizado.

**--post_training_quantize.** Tipo: booleano. Valor predeterminado: falso. Booleano que indica si cuantificar los pesos del modelo flotante convertido. El tamaño del modelo se reducirá y habrá mejoras de latencia (a costa de la precisión).
Para convertir a kmodel recuerda que en la carpeta images:\ de la herramienta necesitamos meter algunas imágenes, en este caso he usado las mismas del conjunto de datos del entrenamiento  asegurándome que tiene una resolución 224x224(las puedes cambiar de resolución si no tuvieran el tamaño deseado).

./tflite2kmodel.sh workspace/mbnet75.tflite

![](../master/imagenes/Image%20%5B14%5D.png)



Con esto ya tendríamos nuestro modelo en formato kmodel y listo para probarlo.

# 7. Introducir el kmodel usando Micropython con MaixpyIde o C con Standalone o FreeRTOS:

Herramienta para introducir todos los archivos necesarios:

https://github.com/sipeed/kflash_gui/releases/

+ MaixpyIde:

Descargar el ide:

http://dl.sipeed.com/MAIX/MaixPy/ide/_/v0.2.4/

Añadir firmware de maixpy he usado esta versión:

Nota: Para que funcione correctamente es necesario tener un firmware igual o superior a al versión 0.4

http://dl.sipeed.com/MAIX/MaixPy/release/master/maixpy_v0.5.0_19_g64e411a/

Añadir el modelo en la dirección 0x00200000

Introducir en la memoria flash del sistema el archivo labels.txt que contiene los “synsets” correspondientes al conjunto de datos (lista de palabras o frases que pueden sustituirse entre ellas en algún contexto).

Aquí se puede descargar el archivo:

https://en.bbs.sipeed.com/uploads/default/original/1X/d41ad9dfbe01f228abe726986fbf1baf4e288f2e.zip

Para hacerlo podemos usar distintas opciones, yo he usado uPyloader, esta información se puede encontrar en:

https://maixpy.sipeed.com/en/get_started/edit_file.html

Una vez que tengamos todo solo necesitamos ejecutar la demo en micropython.

Standalone o FreeRTOS:

En mi caso he usado Standalone.

He usado platformIO como ide para compilar el proyecto, obteniendo el firmware que introduciremos en la dirección 0x00000000 y el modelo en la dirección 0x00200000.

http://blog.sipeed.com/p/622.html

https://docs.platformio.org/en/latest/boards/kendryte210/sipeed-maix-go.html

https://docs.platformio.org/en/latest/platforms/kendryte210.html


## Gráficas y sus limitaciones

![](../master/imagenes/Image%20%5B15%5D.png)

![](../master/imagenes/Image%20%5B16%5D.png)

![](../master/imagenes/Image%20%5B17%5D.png)


Limitaciones:

K210 dispone de 6 MB SRAM para propósito general (según la documentación usables entre 5.5 y 5.9) y 2MB on-chip AI SRAM solamente para el KPU.

Por lo que podemos decir que alrededor de 5 MB disponibles en un entorno C mientras que al usar el entorno de MicroPython tendremos entre 3.5MB y 4MB.

La memoria flash tiene 16MB estando el firmware desde la dirección 0x00000000.



