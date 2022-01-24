# Fast-Layers
Fast-Layers is a python library for Keras and Tensorflow users: The fastest way to build complex deep neural network architectures with sequential models

Installation: !pip install fast-layers

https://pypi.org/project/fast-layers/

## Introduction
Tensorflow's sequential model is a very intuitive way to start learning about Deep Neural Networks.
However it is quite hard to dive into more complex networks without learning more about Keras.

Well it won't be hard anymore with Fast-layers! Define your Sequences and start building complex layers in a sequential fashion.

I created fast-layers for beginners who wants to build more advanced networks and for experimented users who needs to quickly build and test complex module architectures.

# Documentation

    Please note eager execution is not supported yet

#### class Sequence:
    Arguments:
        name: str, positional arg
        inputs: str: name of input pipe/connector | list: names of input pipes/connectors, positional arg
        sequence=None: list of keras.layers objects,
        is_output_layer=False,
        trainable=True,

    Attributes:
        inputs: str or list of input names.
        sequence: list of keras.layers objects,
        is_output_layer: True if this is the output Sequence of a Layer object.
        
    Methods:
        call(x, training=False): by calling the sequence through __call__(), computes x.
        self_build(): build the layers of the sequence into this Sequence object.


#### class Layer:
    Arguments:
        sequences: list of sequences,
        trainable=True,
        n_iteration_error=50: max number of iteration permitted in the computation loop before break

    Attributes:
        names: list of sequences names
        trainable: True if the weights of this layer are trainable.
        sequences: list of sequences
        first_call=True: False means the Layer object has been called and
        n_iteration_error: max number of iteration permitted in the computation loop before break

    Methods:
        init_layer(sequences): Takes a list of sequences and initialize the layer. Is called on __init__() if the layer
                               object has been instantiate with the argument sequences=*List of sequences*
        call(x, training=False): by calling the layer through __call__(), computes x.


## TUTORIAL: MNIST classification using Inception modules with Fast-Layers

### Try it yourself: https://www.kaggle.com/alexandremahdhaoui/fast-layers-tutorial !


original MNIST tutorial: https://www.tensorflow.org/datasets/keras_example

Szegedy et al. 2014, Going deeper with convolutions: https://arxiv.org/pdf/1409.4842.pdf!

![szegedy et al 2014 Inception Module](https://user-images.githubusercontent.com/80970827/112069667-863ff780-8b6c-11eb-8c90-52c3cbc7917a.png)


```python
# Imports and preprocessing
import fast_layers as fl
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.batch(128)
ds_test = ds_test.batch(128)
```

```python
N_FILTERS = 16
PADDING = 'same'

inception_module = fl.Layer()
sequences = [
    fl.Sequence('c1', 'input', sequence = [
        tf.keras.layers.Conv2D(N_FILTERS, (1,1), padding=PADDING)
    ]),
    fl.Sequence('c1_c3', 'input', sequence = [
        tf.keras.layers.Conv2D(N_FILTERS, (1,1), padding=PADDING),
        tf.keras.layers.Conv2D(N_FILTERS, (3,3), padding=PADDING)
    ]),
    fl.Sequence('c1_c5', 'input', sequence = [
        tf.keras.layers.Conv2D(N_FILTERS, (1,1), padding=PADDING),
        tf.keras.layers.Conv2D(N_FILTERS, (5,5), padding=PADDING)
    ]),
    fl.Sequence('maxpool3_c1', 'input', sequence = [
        tf.keras.layers.Conv2D(N_FILTERS, (3,3), padding=PADDING),
        tf.keras.layers.Conv2D(N_FILTERS, (1,1), padding=PADDING)
    ]),
    fl.Sequence('concat', ['c1','c1_c3','c1_c5','maxpool3_c1'], 
             is_output_layer=True,
             sequence=[
                 tf.keras.layers.Concatenate(axis=-1)])
]
inception_module.init_layer(sequences)
```

```python
# A Layer can also be called like this:
sequences_2 = [
    fl.Sequence('c1', 'input', sequence = [
        tf.keras.layers.Conv2D(N_FILTERS, (1,1), padding=PADDING)
    ]),
    fl.Sequence('c1_c3', 'input', sequence = [
        tf.keras.layers.Conv2D(N_FILTERS, (1,1), padding=PADDING),
        tf.keras.layers.Conv2D(N_FILTERS, (3,3), padding=PADDING)
    ]),
    fl.Sequence('c1_c5', 'input', sequence = [
        tf.keras.layers.Conv2D(N_FILTERS, (1,1), padding=PADDING),
        tf.keras.layers.Conv2D(N_FILTERS, (5,5), padding=PADDING)
    ]),
    fl.Sequence('maxpool3_c1', 'input', sequence = [
        tf.keras.layers.Conv2D(N_FILTERS, (3,3), padding=PADDING),
        tf.keras.layers.Conv2D(N_FILTERS, (1,1), padding=PADDING)
    ]),
    fl.Sequence('concat', ['c1','c1_c3','c1_c5','maxpool3_c1'], 
             is_output_layer=True,
             sequence=[
                 tf.keras.layers.Concatenate(axis=-1)])
]


inception_module_2 = fl.Layer(sequence = sequences_2)

```

```python
# Create and train the model
model = tf.keras.models.Sequential([
    inception_module,
    inception_module_2,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)


history = model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
    verbose=2
)

```

