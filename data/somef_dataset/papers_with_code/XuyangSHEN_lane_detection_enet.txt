# Deep Neural Network for Lane Detection  
Lane detection: a semantic segmentation approach.  
Use tensorflow to implement a deep neural network for lane detection based on the paper “ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation.” You can refer this for more details: https://arxiv.org/pdf/1606.02147v1.pdf ​. This model is an encoder-decoder network for general road scene semantic segmentation. We modify that to suit for out task.

* Notes: The coding environment is `python3`. Please install correct python interrupter (we suggest the anaconda environment) and follow up the instructions to run the training or prediction.

## Installation
* This software has only been tested on Win10(64-bit), python3.7.1, Cuda-10.0 with a GTX-1060 6GB GPU.
* Currently, our codes can only run on the tensorflow-gpu, which means it may raise exceptions if the program runs at CPU

Please install all the required package in your environment: (we also provide the `conda bash` command below)
> tensorflow: v1.13.1 `conda install -c conda-forge tensorflow`  
> tensorflow-gpu: v1.13.1 `conda install -c anaconda tensorflow-gpu`  
> tensorboard: v1.13.1 `conda install -c conda-forge tensorboard`  
> opencv: v3.4.1 or v4.1.0 `conda install -c conda-forge opencv`  
> Argparse: v1.3.0 `conda install -c anaconda argparse`  
> imageio: v1.5.0 `conda install -c menpo imageio`  

## Train Model
Caused our model is designed based on the TuSimple lane detection competition, we only tested the dataset provided by the TuSimple.
[The download URL for the whole dataset can be referred here](https://github.com/TuSimple/tusimple-benchmark/issues/3). 
* Note: Due to the offline generator(for train labels), it is a bit tricky(complex) to guide the training instructions. To make  it clear, we provide two main methods to train the model.
The first main method only requires to download the TuSimple train set, while the second one requires to download another training component `label_set-tfRecord` [Download Address](https://github.com/XuyangSHEN/lane_detection_enet/raw/master/train_set/labels.zip).

* training commands:
```bash
$ python training.py -h
usage: training.py [-h] [-i INPUT-PATH] [-m MODEL-PATH] [-l yes/no] [-gt PATH]
                   [-e iteration] [-b per iteration] [-f buffer size]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT-PATH         The path to the train_set (include the train_set).
                        (default: sample_data/train_set/)
  -m MODEL-PATH         The path to store the training model. (default:
                        tmp/train_01/)
  -l yes/no             if it is the first time to run the train, please
                        select yes [offline generator]
  -gt PATH, --ground_truth PATH
                        input the json file of the labels
  -e iteration, --epochs iteration
                        input the number of epochs need to run (default: 40)
  -b per iteration, --batch_size per iteration
                        input the batch size (default: 4)
  -f buffer size, --buffer_size buffer size
                        input the buffer size (default: 150)

```

### to run a small training
Apart from the large training set, we also provide one sample training set to run a small training.

```bash
simple command: 
$ python training.py
full command:
$ python training.py -i  sample_data/train_set/ -m tmp/train_01/ -l no
```

### Main Method1: run training with TuSimple Dataset
Requirement:
* Download the train set: [The URL for the whole dataset can be referred here](https://github.com/TuSimple/tusimple-benchmark/issues/3)
* Please select one Json to train(among the `_0301`, `_0501`, `_0601`). We also kindly provide the combination Json to train all of them. [JSON Download URL](https://github.com/XuyangSHEN/lane_detection_enet/raw/master/train_set/total_label.zip)
* Please check whether the `xxxlabel.Json` is inside the `train_set` folder

Sample command:
```bash
$ python training.py -i (data set path) -m (model storage path) -l yes -gt (xxx.json) 
```

### Main Method2: run training with TuSimple Dataset plus the downloaded tfRecord
Requirement:
* Download the train set:  [The URL for the whole dataset can be referred here](https://github.com/TuSimple/tusimple-benchmark/issues/3)
* Download the labels represented by tfRecord: [The URL for the whole label set can be referred  here](https://github.com/XuyangSHEN/lane_detection_enet/raw/master/train_set/labels.zip)
* Download the combination Json file: [JSON Download URL](https://github.com/XuyangSHEN/lane_detection_enet/raw/master/train_set/total_label.zip)
* Move the downloaded `label` folder into the `train_set` folder
* Please check whether the `xxxlabel.Json` is inside the `train_set` folder

Sample command:
```bash
$ python training.py -i (data set path) -m (model storage path) -l no
```

## Predict Model
We have uploaded the pre-trained model inside the `pre_trained_model`folder, which is trained based on [tusimple benchmark](http://benchmark.tusimple.ai/#/).
You can also follow the `Train Model` instruction to train your own model and make a prediction.

* Please download the Pre-trained model before run the prediction [pre_trained_model](https://drive.google.com/file/d/1GMlc6KFTYyubuj6n9zGhhniW1Q8At7oH/view?usp=sharing)

* prediction commands:
```bash
$ python prediction.py -h
usage: prediction.py [-h] [-i INPUT-PATH] [-o OUTPUT-PATH] [-m MODEL-PATH]
                     [-d DISPLAY]

optional arguments:
  -h, --help      show this help message and exit.
  -i INPUT-PATH   The path to the prediction data folder. [Default: 'sample_data/test_set/']
  -o OUTPUT-PATH  The path to put the prediction result folder. [Default:'tmp_pred/']
  -m MODEL-PATH   The path to the model storage. [Default: 'pre_trained_model/']
  -d DISPLAY      whether to display prediction result or not. [Options: yes/no, Default: no]

```
* sample command:
```bash
$ python prediction.py -i sample_data/test_set/ -o tmp_pred/prediction01/ -m pre_trained_model/ -d no
```

## Contents
* `data_provider` folder:  
> data_dataset.py: provide data for training. tf.Dataset version  
> data_np.py:provide data for training. numpy version  
> label2tfrecord.py: offline label generator  
> line_connect.py: to fix the gaps for the ground truth  
> one_hot_coding.py: one hot labeling (differ from tf.one_hot)  

* `enet` folder:
> ENet.py: netural network layers 
> ENet_Components: components for the ENet model

* `main root:`
> config.py: command line parser  
> training.py: to train the ENet  
> prediction.py: to predict the result  

* `pre_trained_model` folder:
> store the pre-trained model

* `sample_data` folder:
> including some sample data and results

## Result demo:
* Video: [Link](https://youtu.be/tDCkRfYBk4U)

## Members:
**Alisdair Cameron**, **Xinqi ZHU**, **Xuyang SHEN**

## Reference:
* [ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation](https://arxiv.org/pdf/1606.02147v1.pdf)
* [TensorFlow-ENet](https://github.com/kwotsin/TensorFlow-ENet)
* [Implementation of Max Unpooling](https://github.com/tensorflow/tensorflow/issues/2169)


## Feedback:
We are happy to receive any feedback or bug report. Please email me: xuyangshen@yahoo.com
