# Introduction

Simple image classification project using Tensorflow. It allows to train DNN model
and classify images. Contains pretrained models for cat/dog/human recognition.


# Prerequistions

Project was developed using Linux and it is recommended target OS.

Training was performed with Tensorflow 2.1.0. But for simplicity Tensorflow was
used with Docker. So the only true requirement is Docker itself.
For docker installation refer to:
https://docs.docker.com/engine/install/. In case of problems there are some
useful post-installation steps:
https://docs.docker.com/engine/install/linux-postinstall/.

About Tensorflow run in docker you can be read here:
https://www.tensorflow.org/install/docker.
If you want to use GPU within docker follow this instruction:
https://github.com/NVIDIA/nvidia-docker.

Note: file requirements.txt is generated for tensorflow docker image! 


<div id="data"></div>

# Train data

Data set for pretrained models is not available, since it was delivered as part of
closed task from outside source. Data organization for training is as follows:

```
data
└───train
    └───category1
    │       file1.jpg
    │       file2.jpg
    │       ...
    └───category2
    │       file3.jpg
    │       file4.jpg
    │       ...
    └───category3
    │       file5.jpg
    │       file6.jpg
    │       ...   
``` 

where "category1", "category", "category3" are respectively names of data classes.

Original data consisted of:
* 12000 files of cats
* 11000 files of dogs
* 12232 files of human faces

Data was split and 80% was used for training, 10% for validation and 10% for test.


# Usage

To start working with tensorflow in docker run:

```
./scripts/run_in_docker.sh bash
```

or if you can work with GPU:

```
./scripts/run_in_docker_gpu.sh bash
```


## Training

To train new image classification model run:

```
./scripts/run_in_docker.sh python src/train.py --arch vgg_v2
```

There are 3 predefined model architectures (--arch arg) available:

* baseline - simple convolutional network;
* vgg_v1 - VGG like architecture small version;
* vgg_v2 - VGG architecture bigger version.

For the idea behind VGG network architecture refer to:
https://arxiv.org/pdf/1409.1556.pdf.

More training configuration options and parameters can be found with:

```
./scripts/run_in_docker.sh python src/train.py -h

``` 
Remember to have training [data](#data) set up.
Training results be default will be located in folder:

```
models/
```

Along with tensorflow model checkpoint files, there are stored additional files:

* model-HASH.conf - contains json with training configurable parameters and is
required for performing evaluation/prediction.
* model-HASH.history - record of training loss and accuracy values, may be used
to visualize training performance.


## Evaluation

To get more detailed evaluation on test set separated from training data, run:

```
./scripts/run_in_docker.sh python src/evaluate.py -m models/model-HASH.mdl
```

or to evaluate all models located in given folder:

```
./scripts/run_in_docker.sh python src/evaluate.py -d models/
```

Evaluation results will be displayed on standard output:

```
Loading category: cat
Loading category: dog
Loading category: human
Found 3523 validated image filenames belonging to 3 classes.

Model: models/model-2a05ce2.mdl
Total accuracy: 0.9472040874254897
cat accuracy
0.9738396624472574
dog accuracy
0.8554801163918526
human accuracy
0.9954093343534812
```

## Predefined model

Before using, predefined model has to be unpacked with command:

```
cat models/model-2a05ce2.mdl.data-00001-of-00002.tar.gz.* | tar xzv
```



## Predictions

To classify jpg images run:

```
./scripts/run_in_docker.sh python src/predict.py -m models/model-2a05ce2.mdl \
        -d data/samples/ -o output_file.csv
```

You may also pass input files to predict as file with list of images [-l list] or
pass separate files as arguments. Results will be stored in csv format in file
passed with option [-o output_file]. Example output file:

```
file_name,dog,cat,human
data/samples/cat0.JPG,0,1,0
data/samples/dog0.jpg,1,0,0
data/samples/human0.jpg,0,0,1
```

# Results


As yet unresolved problem appeared, where predictions and evaluations of same model
perform slightly different on different hosts (although same docker images). Taking
this into considerations scores are presented in two versions: 1st matching
training machine, 2nd matching alternative host in brackets.

Evaluation on test set separated from training data (10% of original data set):

```Model: models/model-2a05ce2.mdl
Total accuracy: 0.9472040874254897 (0.9477717854101618)
cat accuracy:   0.9738396624472574 (0.9755274261603376)
dog accuracy:   0.8554801163918526 (0.8564500484966052)
human accuracy: 0.9954093343534812 (0.9946442234123948)
```

Results were achieved with following training command:

```
/scripts/run_in_docker_gpu.sh python src/train.py -a --width 256 --height 256 \
    --arch vgg_v2 -b 16 -l 0.0001 -e 10
```

Training was performed with GTX 1060 and took around. 13 hours.
