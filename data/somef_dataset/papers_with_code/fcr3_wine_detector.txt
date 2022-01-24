# Tutorials For Using OpenVINO

## Introduction

This respository contains a number of tutorials on how to use OpenVINO. In particular, these tutorials teach how someone would like to get started using OpenVINO through the context of object detection and pose estimation. Feel free to flip through the Jupyter Notebooks in order to understand how OpenVINO's Python API works.

## Featured Tutorials

These tutorials show how to program using OpenVINO's Python API. Some of the sections within each iPython Notebook are not generalizable, i.e. the networks I use accept input and output detections differently then perhaps other networks. These tutorials are meant for you to extrapolate on your own, since you should know your network and what it accepts as input/what it spits out as output.

### Simple Object Detection

Contains a brief example of how to set up a network, load it into a plugin, and run inference.

### Real-time Object Detection

Contains a slightly more complicated example, but puts the pieces together from the simple object detection module in order to detect different kinds of wine from a webcam.

### Async Real-time Object Detection

Basically the same tutorial as the real time object detection one, but using multiple requests to distribute inference tasks. While we wait for the inference task we care about to finish, others will be processing in the background.

### Core API Tutorials

Accompanying the simple object detection tutorial, real-time object detection tutorial, and the async real-time object detection tutorial are also tutorials that pretty much do the same thing but utilize the IECore instead of IEPlugin. IEPlugin may/will be deprecated in future releases of OpenVINO.

### Loading Mutliple Networks (Also features Reshaping of Input Layer/Checking support for Layers)

Contains a not-so-brief but hopefully clear walkthrough on how someone can use multiple networks in an application.

## Dependencies

### Packages/Libraries
- OpenVINO 2019 R2
- OpenCV
- Numpy
- jupyter notebook
- matplotlib

### Networks
Downloading these networks and put them in a folder called `handpose_optimized`. Within `handpose_optimized`, create two folders called `fp32` and `fp16`. Put the contents of `reverse_hand` into `fp32` and the contents of `fp16` into `fp16`.
- Download files from here and put in `./reverse_hand` directory! https://drive.google.com/open?id=1o7ZeLIcNgb5f-gJh6qwnGM0HJ4TzJQ8T
- Download files from here and put in `./reverse_hand_fp16` directory! https://drive.google.com/open?id=1uXejmgdDaVr2bsbFrrVdlk5nAMm6Fq2i


## Conversion from Tensorflow to IR
This assumes that you have set up the OpenVINO environment. Replace `##` with either `16` or `32`.

Linux:
```
$ mo.py \
>> --input_model wine_export/frozen_inference_graph.pb \
>> --tensorflow_use_custom_operations_config ssd_support_api_v1.14.json \
>> --tensorflow_object_detection_api_pipeline_config wine_export/pipeline.config \
>> --data_type FP## \
>> --output_dir wine_optimized/fp##/
```

Windows:
```
C:\Users\<USERNAME>\<INSTALL_DIR>> "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\model_optimizer\mo.py" --input_model wine_export\frozen_inference_graph.pb --tensorflow_use_custom_operations_config ssd_support_api_v1.14.json --tensorflow_object_detection_api_pipeline_config wine_export\pipeline.config --data_type FP16 --output_dir test_dir
```

## Citations

LearOpenCV Website (used for post-processing HandPose):

https://www.learnopencv.com/hand-keypoint-detection-using-deep-learning-and-opencv/

OpenPose Citation:


    @inproceedings{cao2018openpose,
      author = {Zhe Cao and Gines Hidalgo and Tomas Simon and Shih-En Wei and Yaser Sheikh},
      booktitle = {arXiv preprint arXiv:1812.08008},
      title = {Open{P}ose: realtime multi-person 2{D} pose estimation using {P}art {A}ffinity {F}ields},
      year = {2018}
    }

    @inproceedings{cao2017realtime,
      author = {Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
      booktitle = {CVPR},
      title = {Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
      year = {2017}
    }

    @inproceedings{simon2017hand,
      author = {Tomas Simon and Hanbyul Joo and Iain Matthews and Yaser Sheikh},
      booktitle = {CVPR},
      title = {Hand Keypoint Detection in Single Images using Multiview Bootstrapping},
      year = {2017}
    }

    @inproceedings{wei2016cpm,
      author = {Shih-En Wei and Varun Ramakrishna and Takeo Kanade and Yaser Sheikh},
      booktitle = {CVPR},
      title = {Convolutional pose machines},
      year = {2016}
    }

Links to the papers:

- [OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/abs/1812.08008)
- [Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/abs/1611.08050)
- [Hand Keypoint Detection in Single Images using Multiview Bootstrapping](https://arxiv.org/abs/1704.07809)
- [Convolutional Pose Machines](https://arxiv.org/abs/1602.00134)
