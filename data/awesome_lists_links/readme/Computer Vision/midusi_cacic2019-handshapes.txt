# cacic2019-handshapes
Experiments for the article "Handshape Recognition for Small Dataset" 

## Content

- [Quickstart](#quickstart)
- [Datasets](#datasets)
- [Models and Techniques](#models-&-techniques)
  - [Prototypical Networks for Few-shot Learning](#prototypical-networks-for-few-shot-learning)
  - [Dense Net](#densenet)
  - [Transfer Learning](#transfer-learning)
- [Results](#results)

## Quickstart

To start the docker container execute the following command

```sh
$ ./bin/start [-n <string>] [-t <tag-name>] [--sudo] [--build] [-d] [-c <command>]
```

### Tags

- **latest**	The latest release of TensorFlow CPU binary image. Default.
- **nightly**	Nightly builds of the TensorFlow image. (unstable)
version	Specify the version of the TensorFlow binary image, for example: 2.1.0
- **devel**	Nightly builds of a TensorFlow master development environment. Includes TensorFlow source code.

### Variants

> Each base tag has variants that add or change functionality:

- **\<tag\>-gpu**	The specified tag release with GPU support. (See below)
- **\<tag\>-py3**	The specified tag release with Python 3 support.
- **\<tag\>-jupyter**	The specified tag release with Jupyter (includes TensorFlow tutorial notebooks)

You can use multiple variants at once. For example, the following downloads TensorFlow release images to your machine. For example:

```sh
$ ./bin/start -n myContainer --build  # latest stable release
$ ./bin/start -n myContainer --build -t devel-gpu # nightly dev release w/ GPU support
$ ./bin/start -n myContainer --build -t latest-gpu-jupyter # latest release w/ GPU support and Jupyter
```

Once the docker container is running it will execute the contents of the /bin/execute file.

You can execute

```sh
$ docker exec -it <container-id> /bin/sh -c "[ -e /bin/bash ] && /bin/bash || /bin/sh"
```
to access the running container's shell.

## Datasets

In our paper we used the datasets RWTH-Phoenix, LSA16 and CIARP. We used the library (https://github.com/midusi/handshape_datasets) to fetch the datasets.

## Models & Techniques

### Prototypical Networks for Few-shot Learning

Tensorflow v2 implementation of NIPS 2017 Paper _Prototypical Networks for Few-shot Learning_.

Implementation using [protonet](https://github.com/ulises-jeremias/prototypical-networks-tf).

<details><summary>Training and Eval</summary>

#### Training

Run the following command to run training on `<config>` with default parameters.

```sh
$ ./bin/run --model protonet --mode train --config <config>
```

`<config> = lsa16 | rwth | Ciarp`

#### Evaluating

To run evaluation on a specific dataset

```sh
$ ./bin/run --model protonet --mode eval --config <config>
```

`<config> = lsa16 | rwth | Ciarp`
</details>

### Dense Net

We implemented Densenet using squeeze and excitation layers in tensorflow 2 for our experiments. To see its implementation go to [densenet](https://github.com/okason97/DenseNet-Tensorflow2).

For more information about densenet please refer to the [original paper](https://arxiv.org/abs/1608.06993).

<details><summary>Training and Eval</summary>

#### Training

Run the following command to run training on `<config>` with default parameters.

```sh
$ ./bin/run --model densenet --mode train --config <config>
```

`<config> = lsa16 | rwth | Ciarp`

#### Evaluating

To run evaluation on a specific dataset

```sh
$ ./bin/run --model densenet --mode eval --config <config>
```

`<config> = lsa16 | rwth | Ciarp`
</details>

### Transfer Learning

<details><summary>Training and Eval</summary>

#### Training

Run the following command to run training on `<config>` with default parameters.

```sh
$ ./bin/run --tl --model <model> --mode train --config <config>
```

```
<model> = vgg16 | vgg19 | inception_v3 | densenet | densenet169 | densenet201
<config> = lsa16 | rwth | Ciarp
```
#### Evaluating

To run evaluation on a specific dataset

```sh
$ ./bin/run --tl --model <model> --mode eval --config <config>
```

```
<model> = vgg16 | vgg19 | inception_v3 | densenet | densenet169 | densenet201
<config> = lsa16 | rwth | Ciarp
```
</details>

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
│  │  └─ └─ <dataset>_<model>_results.csv  # ".csv" file in which the relationships between configurations, models, results and 
summaries are listed by date.
│  └─ summary.csv                          # contains the summary of all the training
└─ . . .
```

where

```
<dataset> = lsa16 | rwth | Ciarp
<model> = densenet | protonet | vgg16 | vgg19 | inception_v3 | densenet | densenet169 | densenet201
```

To run TensorBoard, use the following command:

```sh
$ tensorboard --logdir=./results/<dataset>/<model>/summaries
```
