# Keras Custom Layers of AdaCos and ArcFace

Keras Custom Layers of AdaCos and ArcFace, and experiments in caltech birds 2011(CUB-200-2011).

## Original Paper

* AdaCos: Adaptively Scaling Cosine Logits for Effectively Learning Deep Face Representations<br>
    [https://arxiv.org/abs/1905.00292](https://arxiv.org/abs/1905.00292)
* ArcFace: Additive Angular Margin Loss for Deep Face Recognition<br>
    [https://arxiv.org/abs/1801.07698](https://arxiv.org/abs/1801.07698)
* L2-constrained Softmax Loss for Discriminative Face Verification<br>
    [https://arxiv.org/abs/1703.09507](https://arxiv.org/abs/1703.09507)

## Building Model sample by the Functional API

```python
num_classes = 200 # CUB-200-2011
img_size = 224    # EfficientNetB0

from CustomLayer import AdaCos
feature_extractor_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1", name='efficientnetB0')
input_image = tf.keras.Input(shape=(img_size, img_size, 3), dtype=tf.float32, name='input_image')
efficientnet_output = feature_extractor_layer(input_image)

cos_layer = CosineLayer(num_classes=num_classes)
cos_layer_output = cos_layer(efficientnet_output)

logits = AdaCos_logits()([cos_layer_output, y_true])

model = tf.keras.models.Model(inputs=(input_image, y_true), outputs=tf.keras.layers.Softmax()(logits)
```

<img src="docs/img/model.png">

## Requirements

* tensorflow > 2.2
* tensorflow_probability

## Experiments in caltech birds 2011(CUB-200-2011)

* Using data

    caltech_birds tfds ([https://www.tensorflow.org/datasets/catalog/caltech_birds2011](https://www.tensorflow.org/datasets/catalog/caltech_birds2011)).

* Preprocess

    crop to bounding box, general augmentation(flip, brightness, rotate, etc.), resize to [224,224,3] (EfficientnetB0) and return ((image, label), label) for using label to train

    ```shell-session
    >>>train_batches
    <MapDataset shapes: (((None, None, None, 3), (None, 1)), (None,)), types: ((tf.float32, tf.int64), tf.int64)>
    ```

* Model

    [EfficientNetB0](https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1) with AdaCos, fixedAdaCos, ArcFace, l2-softmax and softmax

* Result

  * Accuracy of test data is almost same(0.82) except for softmax(0.8).

  * AdaCos value of s gradually decreases as with the paper.

  * Average *cos&theta;* of correct label increases to near 1(*&theta;* = 0) in AdaCos and fixedAdaCos.

  <img src="docs/img/result.png">
