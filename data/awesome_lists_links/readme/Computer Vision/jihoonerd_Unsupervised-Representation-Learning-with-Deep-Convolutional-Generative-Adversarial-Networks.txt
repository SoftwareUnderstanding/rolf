# Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks

Implementation of the paper [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)

## Caveats

* Make sure your `~/.keras/keras.json` uses tensorflow and `float32` for `floatx`!
* Currently, only the dataset in `tf.keras.dataset` are supported.
* If you are using CPU, it will take a very long time. Strongly recommend to use GPU for training. (In this case, you need to install `tensorflow-gpu` instead of `tensorflow` in `requirements.txt`)

## Demo images

|Data|DCGAN|Last Image|
|---|---|---|
|MNIST|![mnist_gif](/assets/mnist.gif)|![last_image](/assets/mnist_images/images_at_epoch_0099.png)|
|CIFAR10|![cifar10_gif](/assets/cifar10.gif)|![광화문2+gogh](/assets/cifar10/images_at_epoch_0299.png)|

## TensorBoard

![mnist_tb](/assets/mnist_tb.png)

## How to run

`--help` option will show necessary arguments.

```bash
$ python main.py --helpusage: main.py [-h] [--data DATA] [--keras_dataset] [--color]
               [--log_interval LOG_INTERVAL] [--epochs EPOCHS]
               [--batch_size BATCH_SIZE]

DCGAN Trainer

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           Dataset name, currently only supports keras dataset
  --keras_dataset       Boolean
  --color
  --log_interval LOG_INTERVAL
  --epochs EPOCHS
  --batch_size BATCH_SIZE
```

## BibTeX
```
@misc{radford2015unsupervised,
  abstract = {In recent years, supervised learning with convolutional networks (CNNs) has
seen huge adoption in computer vision applications. Comparatively, unsupervised
learning with CNNs has received less attention. In this work we hope to help
bridge the gap between the success of CNNs for supervised learning and
unsupervised learning. We introduce a class of CNNs called deep convolutional
generative adversarial networks (DCGANs), that have certain architectural
constraints, and demonstrate that they are a strong candidate for unsupervised
learning. Training on various image datasets, we show convincing evidence that
our deep convolutional adversarial pair learns a hierarchy of representations
from object parts to scenes in both the generator and discriminator.
Additionally, we use the learned features for novel tasks - demonstrating their
applicability as general image representations.},
  added-at = {2018-05-21T17:55:23.000+0200},
  author = {Radford, Alec and Metz, Luke and Chintala, Soumith},
  biburl = {https://www.bibsonomy.org/bibtex/2a114a1bd36bb9b5542f620b0c1d1c050/lw4},
  description = {Unsupervised Representation Learning with Deep Convolutional Generative
  Adversarial Networks},
  interhash = {ae6fc4b7593a1d0e31aeeff9fef81a36},
  intrahash = {a114a1bd36bb9b5542f620b0c1d1c050},
  keywords = {gan},
  note = {cite arxiv:1511.06434Comment: Under review as a conference paper at ICLR 2016},
  timestamp = {2018-05-21T17:55:23.000+0200},
  title = {Unsupervised Representation Learning with Deep Convolutional Generative
  Adversarial Networks},
  url = {http://arxiv.org/abs/1511.06434},
  year = 2015
}
```
