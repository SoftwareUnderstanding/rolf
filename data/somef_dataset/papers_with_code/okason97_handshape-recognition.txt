# Handshape Recognition

## Content

- [Results](#results)
- [Quickstart](#quickstart)
- [Setup and use docker](#setup-and-use-docker)
- [Models](#models)
  - [Prototypical Networks for Few-shot Learning](#prototypical-networks-for-few-shot-learning)
    - [Evaluating](#evaluating)
  - [Dense Net](#dense-net)

## Results

In the `/results` directory you can find the results of a training processes using a `<model>` on a specific `<dataset>`:

```
.
├─ . . .
├─ results
│  ├─ <dataset>                            # results for an specific dataset.
│  │  ├─ <model>                           # results training a <model> on a <dataset>.
│  │  │  ├─ models                         # ".h5" files for trained models.
│  │  │  ├─ results                        # ".csv" files with the different metrics for each training period.
│  │  │  ├─ summaries                      # tensorboard summaries.
│  │  │  ├─ config                         # optional configuration files.
│  └─ └─ └─ <dataset>_<model>_results.csv  # ".csv" file in which the relationships between configurations, models, results and summaries are listed by date.
└─ . . .
```

where

```
<dataset> = lsa16 | rwth | . . .
<model> = dense-net | proto-net
```

To run TensorBoard, use the following command:

```sh
$ tensorboard --logdir=./results/<dataset>/<model>/summaries
```

* * *

## Quickstart

```sh
$ ./bin/start [-n <string>] [-t <tag-name>] [--sudo] [--build]
```

```
<tag-name> = cpu | devel-cpu | gpu
```

## Setup and use docker

Build the docker image,

```sh
$ docker build --rm -f dockerfiles/tf-py3-jupiter.Dockerfile -t handshape-recognition:latest .
```

and now run the image

```sh
$ docker run --rm -u $(id -u):$(id -g) -p 6006:6006 -p 8888:8888 handshape-recognition:latest
```

Visit that link, hey look your jupyter notebooks are ready to be created.

If you want, you can attach a shell to the running container

```sh
$ docker exec -it <container-id> /bin/sh -c "[ -e /bin/bash ] && /bin/bash || /bin/sh"
```

And then you can find the entire source code in `/develop`.

```sh
$ cd /develop
```

To run TensorBoard, use the following command (alternatively python -m tensorboard.main)

```sh
$ tensorboard --logdir=/path/to/summaries
```

## Models

### Prototypical Networks for Few-shot Learning

Tensorflow v2 implementation of NIPS 2017 Paper _Prototypical Networks for Few-shot Learning_.

Implementation based on [protonet](https://github.com/ulises-jeremias/prototypical-networks-tf).

Run the following command to run training on `<config>` with default parameters.

```sh
$ ./bin/protonet --mode train --config <config>
```

`<config> = lsa16 | rwth`

#### Evaluating

To run evaluation on a specific dataset

```sh
$ ./bin/protonet --mode eval --config <config>
```

`<config> = lsa16 | rwth, rwth not working yet`

### Dense Net

Tensorflow 2 implementation of Densenet using Squeeze and Excitation layers.

Inspired by flyyufelix keras implementation (https://github.com/flyyufelix/DenseNet-Keras).

For more information about densenet please refer to the original paper (https://arxiv.org/abs/1608.06993).

To train run the following command
```sh
$ python train_single.py
```

you can include the following arguments for further customization

Dataset:
```sh
--dataset=<dataset>
```
`<dataset> = lsa16 | rwth`

Rotation angle in degrees:
```sh
--rotation=<int>
```

Widht shift:
```sh
--w-shift=<float>
```

Height shift:
```sh
--h-shift=<float>
```

Horizontal flip:
```sh
--h-flip=<boolean>
```

Densenet's growth rate:
```sh
--growth-r=<int>
```

Densenet's number of dense layers:
```sh
--nb-layers=<nb-layers>
```
`<nb-layers> = <int>[:<int>]*`

Densenet's reduction:
```sh
--reduction=<float>
```

Learning rate:
```sh
--lr=<float>
```

Epochs:
```sh
--epochs=<int>
```

Maximum patience:
```sh
--patience=<int>
```

Log frequency:
```sh
--log-freq=<int>
```

Save frequency (only works if checkpoints is set to True):
```sh
--save-freq=<int>
```

Models directory (only works if checkpoints is set to True):
```sh
--models-dir=<string>
```

Results directory:
```sh
--results_dir=<string>
```

Checkpoint model saving:
```sh
--checkpoints=<boolean>
```

Use of class weights:
```sh
--weight_classes=<boolean>
```
