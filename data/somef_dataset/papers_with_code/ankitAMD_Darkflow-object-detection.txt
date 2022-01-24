## What i perform and Experience.......
I Perfom all the activities like Image ,Video,Webcam detection in Object-Detection using Darkflow on UBNUTU 16.04. I Face Many Problem and changed lot of codes to perform another tasks which i never used and face before.

See my Demo video it is not Accurate detection because of not good dataset , GPU and its corresponding compatible hardware but i used CPU, 8GB-RAM, Intel i7/i5 processor and 1TB Hard-Disk only and perform Custom Object-Detection.

I train my model on 1and 4 classes differently and uses 25 Images of dataset through which i do not get very good performance but i got result to show my demo to Public. I done and implement this project myself without anyhelp except solving some issues from Google, Stackoverflow, Medium.  

##### Edited: 
I have also train on tiny-yolo.cfg and tiny-yolo.weights file on detecting good/bad Solar_panel.But Performance is not Good and threshold is very bad.But it give some results.

https://github.com/ankitAMD/Darkflow-object-detection/blob/master/Custom%20Automated%20Testing%20images%20of%20Solar_Panel%20(using%20tiny-yolo.cfg%20and%20weights).ipynb

###### If you like  Details on this project, might be useful or may be help to clear your doubts and issues , please click on star.

https://github.com/thtrieu/darkflow

https://github.com/ankitAMD/Read-Multiple-images-from-a-folder-using-python-cv2

##### This Video is Demo of the Object-Detection.

https://www.youtube.com/watch?v=-2CkuAU6pRM

https://www.youtube.com/watch?v=pY_ofBTTuBE

##### Inner concept,Short and Better Explanation of this Project..

https://www.analyticsvidhya.com/blog/2018/12/practical-guide-object-detection-yolo-framewor-python/


### Basic Reading and Required  links :

 1. https://github.com/markjay4k/YOLO-series/blob/master/part1%20-%20setup%20YOLO.ipynb

 2. https://www.youtube.com/watch?v=T8wK5loXkXg

 3. https://www.youtube.com/watch?v=RplXYjxgZbw

 4. https://www.lfd.uci.edu/~gohlke/pythonlibs/

 5. https://github.com/thtrieu/darkflow

 6. https://pjreddie.com/darknet/yolo/


### Some Important Commands (used at Testing time).......

For ubuntu users only  video croping command ----

I found ffmpeg could do the task. To install the ffmpeg sudo apt-get install ffmpeg To copy a specific short duration from the video file.

             ffmpeg -i original.mp4 -ss 00:01:52 -c copy -t 00:00:10 output.mp4

  -i:input file. The file name of the original video clip.
  -ss:start timestamp. In my case I tried to copy the new video from the original video started after 1 minute and 52 seconds.
  -t:indicating the duration of the new video clip, in my case, 10 seconds. 
  
   ##### Go to the link for more details
   
            https://askubuntu.com/questions/840982/what-tools-could-i-use-on-ubuntu-16-04-to-cut-mp4-files-into-several-clips/841074

   ##### Video downloading free from youtube online for testing

             http://www.youtube-video-downloader.xyz/

   ##### Changes in Size,Length,Style, Color of Bounding-box and Text
   
   https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/
   
   https://www.javatpoint.com/opencv-drawing-functions
   
   https://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html
   
   https://vovkos.github.io/doxyrest-showcase/opencv/sphinxdoc/enum_cv_HersheyFonts.html


# what is YOLO v2 (aka YOLO 9000)

YOLO9000 is a high speed, real time detection algorithm that can detect on OVER 9000! (object categories)

    you can read more about it here (https://arxiv.org/pdf/1612.08242.pdf)
    watch a talk on it here (https://www.youtube.com/watch?v=NM6lrxy0bxs)
    and another talk here (https://www.youtube.com/watch?v=4eIBisqx9_g)

## Step1 - Requirements
    
    Guidance for installing Anaconda on ubuntu linux (https://docs.anaconda.com/anaconda/install/linux/) this is only i used for reading and following command and installing below Anaconda version(2 line number).
    
    first i install on base terminal (on this step not created any environment)
    
              1.python version 2.7.12(which is already installed with system, check it first)
              
              2.anaconda (https://repo.continuum.io/archive/Anaconda2-4.2.0-Linux-x86_64.sh)(Download conda from this link) 
              
              conda version 4.2.9 (conda --version type on terminal ....no need to install conda separately install above anaconda through link so this conda verion comes with it. )
              
    
    properly install everything on base then i created "abc" environment  following these commands in link ....
      
              (https://github.com/ankitAMD/1Ant_Bees_classification_Pytorch) .
              
              After created abc environment i go into the environemnt "abc".......

               "source activate abc"
              
              in this environment python version is 3.6, check it.
 
              tensorflow version 1.14.0 (pip install tensorflow==1.14)
              pip uninstall tensorflow
              pip install tensorflow==1.4.0
              
              cython version (conda install cython==0.26)
              
              numpy version 1.17.4/1.18 check it  (pip install numpy==1.17.4)
              
              opencv version 4.1.1/4.1.2 (pip install opencv-python)
              
              matplotlib(conda install matplotlib)

              lxml (pip install lxml)
              
    
    Python 3.5 or 3.6. Anaconda (install tutorial https://www.youtube.com/watch?v=T8wK5loXkXg)
    
    Tensorflow (tutorial GPU verions https://www.youtube.com/watch?v=RplXYjxgZbw&t=91s)
    
    openCV (https://www.lfd.uci.edu/~gohlke/pythonlibs/)



   #### Some other way for Installing opencv link ........

1. https://medium.com/analytics-vidhya/installation-of-opencv-in-simple-and-easy-way-15556edca7a4

2. https://medium.com/analytics-vidhya/what-and-why-opencv-3b807ade73a0

3. http://www.codebind.com/cpp-tutorial/install-opencv-ubuntu-cpp/

#### Practice Pytorch from this website........................

1. https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_matrices/

              
## Step2 - Download the Darkflow repo

    https://github.com/thtrieu/darkflow
    
    extract the files somewhere locally

## Step3 - Build the library

    open an cmd window and type

python setup.py build_ext --inplace

OR

pip install -e .

## Step 4 - Download a weights file

    Download the YOLOv2 608x608 weights file here (https://pjreddie.com/darknet/yolov2/)
    NOTE: there are other weights files you can try if you like
    create a bin folder within the darkflow-master folder
    put the weights file in the bin folder

Processing a video file

    move the video file into the ``darkflow-master```
    from there, open a cmd window
    use the command

python flow --model cfg/yolo.cfg --load bin/yolov2.weights --demo videofile.mp4 --gpu 1.0 --saveVideo

videofile.mp4 is the name of your video.

NOTE: if you do not have the GPU version of tensorflow, leave off the --gpu 1.0

--saveVideo indicates to save a name video file, which has the boxes around objects





## Intro

[![Build Status](https://travis-ci.org/thtrieu/darkflow.svg?branch=master)](https://travis-ci.org/thtrieu/darkflow) [![codecov](https://codecov.io/gh/thtrieu/darkflow/branch/master/graph/badge.svg)](https://codecov.io/gh/thtrieu/darkflow)

Real-time object detection and classification. Paper: [version 1](https://arxiv.org/pdf/1506.02640.pdf), [version 2](https://arxiv.org/pdf/1612.08242.pdf).

Read more about YOLO (in darknet) and download weight files [here](http://pjreddie.com/darknet/yolo/). In case the weight file cannot be found, I uploaded some of mine [here](https://drive.google.com/drive/folders/0B1tW_VtY7onidEwyQ2FtQVplWEU), which include `yolo-full` and `yolo-tiny` of v1.0, `tiny-yolo-v1.1` of v1.1 and `yolo`, `tiny-yolo-voc` of v2.


See demo below or see on [this imgur](http://i.imgur.com/EyZZKAA.gif)

<p align="center"> <img src="demo.gif"/> </p>

## Dependencies

Python3, tensorflow 1.0, numpy, opencv 3.

### Getting started

You can choose _one_ of the following three ways to get started with darkflow.

1. Just build the Cython extensions in place. NOTE: If installing this way you will have to use `./flow` in the cloned darkflow directory instead of `flow` as darkflow is not installed globally.
    ```
    python3 setup.py build_ext --inplace
    ```

2. Let pip install darkflow globally in dev mode (still globally accessible, but changes to the code immediately take effect)
    ```
    pip install -e .
    ```

3. Install with pip globally
    ```
    pip install .
    ```

## Update

**Android demo on Tensorflow's** [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/android/src/org/tensorflow/demo/TensorFlowYoloDetector.java)

**I am looking for help:**
 - `help wanted` labels in issue track

## Parsing the annotations

Skip this if you are not training or fine-tuning anything (you simply want to forward flow a trained net)

For example, if you want to work with only 3 classes `tvmonitor`, `person`, `pottedplant`; edit `labels.txt` as follows

```
tvmonitor
person
pottedplant
```

And that's it. `darkflow` will take care of the rest. You can also set darkflow to load from a custom labels file with the `--labels` flag (i.e. `--labels myOtherLabelsFile.txt`). This can be helpful when working with multiple models with different sets of output labels. When this flag is not set, darkflow will load from `labels.txt` by default (unless you are using one of the recognized `.cfg` files designed for the COCO or VOC dataset - then the labels file will be ignored and the COCO or VOC labels will be loaded).

## Design the net

Skip this if you are working with one of the original configurations since they are already there. Otherwise, see the following example:

```python
...

[convolutional]
batch_normalize = 1
size = 3
stride = 1
pad = 1
activation = leaky

[maxpool]

[connected]
output = 4096
activation = linear

...
```

## Flowing the graph using `flow`

```bash
# Have a look at its options
flow --h
```

First, let's take a closer look at one of a very useful option `--load`

```bash
# 1. Load tiny-yolo.weights
flow --model cfg/tiny-yolo.cfg --load bin/tiny-yolo.weights

# 2. To completely initialize a model, leave the --load option
flow --model cfg/yolo-new.cfg

# 3. It is useful to reuse the first identical layers of tiny for `yolo-new`
flow --model cfg/yolo-new.cfg --load bin/tiny-yolo.weights
# this will print out which layers are reused, which are initialized
```

All input images from default folder `sample_img/` are flowed through the net and predictions are put in `sample_img/out/`. We can always specify more parameters for such forward passes, such as detection threshold, batch size, images folder, etc.

```bash
# Forward all images in sample_img/ using tiny yolo and 100% GPU usage
flow --imgdir sample_img/ --model cfg/tiny-yolo.cfg --load bin/tiny-yolo.weights --gpu 1.0
```
json output can be generated with descriptions of the pixel location of each bounding box and the pixel location. Each prediction is stored in the `sample_img/out` folder by default. An example json array is shown below.
```bash
# Forward all images in sample_img/ using tiny yolo and JSON output.
flow --imgdir sample_img/ --model cfg/tiny-yolo.cfg --load bin/tiny-yolo.weights --json
```
JSON output:
```json
[{"label":"person", "confidence": 0.56, "topleft": {"x": 184, "y": 101}, "bottomright": {"x": 274, "y": 382}},
{"label": "dog", "confidence": 0.32, "topleft": {"x": 71, "y": 263}, "bottomright": {"x": 193, "y": 353}},
{"label": "horse", "confidence": 0.76, "topleft": {"x": 412, "y": 109}, "bottomright": {"x": 592,"y": 337}}]
```
 - label: self explanatory
 - confidence: somewhere between 0 and 1 (how confident yolo is about that detection)
 - topleft: pixel coordinate of top left corner of box.
 - bottomright: pixel coordinate of bottom right corner of box.

## Training new model

Training is simple as you only have to add option `--train`. Training set and annotation will be parsed if this is the first time a new configuration is trained. To point to training set and annotations, use option `--dataset` and `--annotation`. A few examples:

```bash
# Initialize yolo-new from yolo-tiny, then train the net on 100% GPU:
flow --model cfg/yolo-new.cfg --load bin/tiny-yolo.weights --train --gpu 1.0

# Completely initialize yolo-new and train it with ADAM optimizer
flow --model cfg/yolo-new.cfg --train --trainer adam
```

During training, the script will occasionally save intermediate results into Tensorflow checkpoints, stored in `ckpt/`. To resume to any checkpoint before performing training/testing, use `--load [checkpoint_num]` option, if `checkpoint_num < 0`, `darkflow` will load the most recent save by parsing `ckpt/checkpoint`.

```bash
# Resume the most recent checkpoint for training
flow --train --model cfg/yolo-new.cfg --load -1

# Test with checkpoint at step 1500
flow --model cfg/yolo-new.cfg --load 1500

# Fine tuning yolo-tiny from the original one
flow --train --model cfg/tiny-yolo.cfg --load bin/tiny-yolo.weights
```

Example of training on Pascal VOC 2007:
```bash
# Download the Pascal VOC dataset:
curl -O https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar

# An example of the Pascal VOC annotation format:
vim VOCdevkit/VOC2007/Annotations/000001.xml

# Train the net on the Pascal dataset:
flow --model cfg/yolo-new.cfg --train --dataset "~/VOCdevkit/VOC2007/JPEGImages" --annotation "~/VOCdevkit/VOC2007/Annotations"
```

### Training on your own dataset

*The steps below assume we want to use tiny YOLO and our dataset has 3 classes*

1. Create a copy of the configuration file `tiny-yolo-voc.cfg` and rename it according to your preference `tiny-yolo-voc-3c.cfg` (It is crucial that you leave the original `tiny-yolo-voc.cfg` file unchanged, see below for explanation).

2. In `tiny-yolo-voc-3c.cfg`, change classes in the [region] layer (the last layer) to the number of classes you are going to train for. In our case, classes are set to 3.
    
    ```python
    ...

    [region]
    anchors = 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52
    bias_match=1
    classes=3
    coords=4
    num=5
    softmax=1
    
    ...
    ```

3. In `tiny-yolo-voc-3c.cfg`, change filters in the [convolutional] layer (the second to last layer) to num * (classes + 5). In our case, num is 5 and classes are 3 so 5 * (3 + 5) = 40 therefore filters are set to 40.
    
    ```python
    ...

    [convolutional]
    size=1
    stride=1
    pad=1
    filters=40
    activation=linear

    [region]
    anchors = 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52
    
    ...
    ```

4. Change `labels.txt` to include the label(s) you want to train on (number of labels should be the same as the number of classes you set in `tiny-yolo-voc-3c.cfg` file). In our case, `labels.txt` will contain 3 labels.

    ```
    label1
    label2
    label3
    ```
5. Reference the `tiny-yolo-voc-3c.cfg` model when you train.

    `flow --model cfg/tiny-yolo-voc-3c.cfg --load bin/tiny-yolo-voc.weights --train --annotation train/Annotations --dataset train/Images`


* Why should I leave the original `tiny-yolo-voc.cfg` file unchanged?
    
    When darkflow sees you are loading `tiny-yolo-voc.weights` it will look for `tiny-yolo-voc.cfg` in your cfg/ folder and compare that configuration file to the new one you have set with `--model cfg/tiny-yolo-voc-3c.cfg`. In this case, every layer will have the same exact number of weights except for the last two, so it will load the weights into all layers up to the last two because they now contain different number of weights.


## Camera/video file demo

For a demo that entirely runs on the CPU:

```bash
flow --model cfg/yolo-new.cfg --load bin/yolo-new.weights --demo videofile.avi
```

For a demo that runs 100% on the GPU:

```bash
flow --model cfg/yolo-new.cfg --load bin/yolo-new.weights --demo videofile.avi --gpu 1.0
```

To use your webcam/camera, simply replace `videofile.avi` with keyword `camera`.

To save a video with predicted bounding box, add `--saveVideo` option.

## Using darkflow from another python application

Please note that `return_predict(img)` must take an `numpy.ndarray`. Your image must be loaded beforehand and passed to `return_predict(img)`. Passing the file path won't work.

Result from `return_predict(img)` will be a list of dictionaries representing each detected object's values in the same format as the JSON output listed above.

```python
from darkflow.net.build import TFNet
import cv2

options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}

tfnet = TFNet(options)

imgcv = cv2.imread("./sample_img/sample_dog.jpg")
result = tfnet.return_predict(imgcv)
print(result)
```


## Save the built graph to a protobuf file (`.pb`)

```bash
## Saving the lastest checkpoint to protobuf file
flow --model cfg/yolo-new.cfg --load -1 --savepb

## Saving graph and weights to protobuf file
flow --model cfg/yolo.cfg --load bin/yolo.weights --savepb
```
When saving the `.pb` file, a `.meta` file will also be generated alongside it. This `.meta` file is a JSON dump of everything in the `meta` dictionary that contains information nessecary for post-processing such as `anchors` and `labels`. This way, everything you need to make predictions from the graph and do post processing is contained in those two files - no need to have the `.cfg` or any labels file tagging along.

The created `.pb` file can be used to migrate the graph to mobile devices (JAVA / C++ / Objective-C++). The name of input tensor and output tensor are respectively `'input'` and `'output'`. For further usage of this protobuf file, please refer to the official documentation of `Tensorflow` on C++ API [_here_](https://www.tensorflow.org/versions/r0.9/api_docs/cc/index.html). To run it on, say, iOS application, simply add the file to Bundle Resources and update the path to this file inside source code.

Also, darkflow supports loading from a `.pb` and `.meta` file for generating predictions (instead of loading from a `.cfg` and checkpoint or `.weights`).
```bash
## Forward images in sample_img for predictions based on protobuf file
flow --pbLoad built_graph/yolo.pb --metaLoad built_graph/yolo.meta --imgdir sample_img/
```
If you'd like to load a `.pb` and `.meta` file when using `return_predict()` you can set the `"pbLoad"` and `"metaLoad"` options in place of the `"model"` and `"load"` options you would normally set.

That's all.
# yolo--darkflow
# credited to https://github.com/thtrieu/darkflow

# Read these issues for better Understanding.................

https://github.com/thtrieu/darkflow/issues/1012 (Can't make a prediction)

https://github.com/thtrieu/darkflow/issues/613 (How do I evaluate accuracy of the test set)

https://github.com/thtrieu/darkflow/issues/638 (Change Bounding Box thickness )

https://github.com/bendidi/Tracking-with-darkflow (Tracking-with-darkflow)

https://github.com/thtrieu/darkflow/issues/717 (Please add the tensorboard for visualising the loss)

https://github.com/thtrieu/darkflow/issues/723  (while training - loss parameter almost doesnt decrease)

https://github.com/thtrieu/darkflow/issues/918 (retrieve loss function values from checkpoints )

https://github.com/thtrieu/darkflow/issues/526  (bigger bbox sizes)

https://github.com/thtrieu/darkflow/issues/1012 (Can't make a prediction)

https://github.com/thtrieu/darkflow/issues/611 (Image confidence show with bounding box in output images)

https://github.com/thtrieu/darkflow/issues/603 (Show training results in tensorboard )

https://github.com/thtrieu/darkflow/issues/566 (Change color of bounding box)

https://github.com/thtrieu/darkflow/issues/283 (how to set confidence threshold)

https://github.com/thtrieu/darkflow/issues/222 (Can't set threshold?)

https://github.com/thtrieu/darkflow/issues/286 (Freezing Graph Issue)

https://github.com/thtrieu/darkflow/issues/222 (Can't set threshold?)

https://github.com/thtrieu/darkflow/issues/190 (--threshold option does not do anything)

https://github.com/thtrieu/darkflow/issues/94  (How could I add tensorboad visualization?)

https://github.com/thtrieu/darkflow/issues/92  (Segmentation fault (core dumped) while running the tiny yolo command as in the README #92)

https://github.com/thtrieu/darkflow/issues/93  (detection threshold not used !!)

https://github.com/thtrieu/darkflow/issues/29 (How to fine-tune?)

https://github.com/thtrieu/darkflow/issues/9  (what is the lowest loss value can reach?)

https://github.com/thtrieu/darkflow/issues/7   (Random scale/translate send object(s) out of bound)



# Some Basic Websites to start to learn ML ..............
           
            http://jakevdp.github.io/blog/2016/08/25/conda-myths-and-misconceptions/
            
            https://github.com/rougier/numpy-100/blob/master/100_Numpy_exercises.ipynb
            
            https://pandas.pydata.org/pandas-docs/version/0.17.0/10min.html#minutes-to-pandas
            
            https://github.com/abhat222/Data-Science--Cheat-Sheet/blob/master/Data%20Science/VIP%20Cheat%20Sheet%20(ML%20%2C%20DL%20%2C%20AI).pdf
            
            https://d2wvfoqc9gyqzf.cloudfront.net/content/uploads/2018/09/Ng-MLY01-13.pdf
            
            https://www.slideshare.net/TessFerrandez/notes-from-coursera-deep-learning-courses-by-andrew-ng?trk
"

