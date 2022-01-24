# Severstal Steel Defect Detection

Steel is one of the most important building materials of modern times. Steel buildings are resistant to natural and man-made wear which has made the material ubiquitous around the world. To help make production of steel more efficient, this competition will help identify defects.

Severstal is leading the charge in efficient steel mining and production. They believe the future of metallurgy requires development across the economic, ecological, and social aspects of the industry—and they take corporate responsibility seriously. The company recently created the country’s largest industrial data lake, with petabytes of data that were previously discarded. Severstal is now looking to machine learning to improve automation, increase efficiency, and maintain high quality in their production.

The production process of flat sheet steel is especially delicate. From heating and rolling, to drying and cutting, several machines touch flat steel by the time it’s ready to ship. Today, Severstal uses images from high frequency cameras to power a defect detection algorithm.

## Automatic Mixed Precision Training

Automatic Mixed Precision (AMP) makes all the required adjustments to train models using mixed precision, providing two benefits over manual operations:
- Developers need not modify network model code, reducing development and maintenance effort.
- Using AMP maintains forward and backward compatibility with all the APIs for defining and running models.

The benefits of mixed precision training are:
- Speed up of math-intensive operations, such as linear and convolution layers, by using Tensor Cores.
- Speed up memory-limited operations by accessing half the bytes compared to single-precision.
- Reduction of memory requirements for training models, enabling larger models or larger minibatches.

Using mixed precision training requires two steps:

1. Porting the model to use the FP16 data type where appropriate.
1. Using loss scaling to preserve small gradient values.

```
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
```

## Neural Network Architecture

[U-net](https://arxiv.org/abs/1505.04597) is used to solve the segmentation problem. U-net model is mostly used for Biomedical use-cases. It follows an encoder-decoder architecture and consists of mainly convolution layers and no-dense layers. It downscales the image to learn the "WHAT" in the images and then upscales the convolution layer to learn the "WHERE" in the images
<p align="center">
<img src = "https://github.com/SarthakGarg13/Severstal-Steel-Defect-Detection/blob/master/images/unet.JPG">
</p>


## TensorRT

The core of NVIDIA® TensorRT is a C++ library that facilitates high-performance inference on NVIDIA graphics processing units (GPUs). It is designed to work in a complementary fashion with training frameworks such as TensorFlow, Caffe, PyTorch, MXNet, etc. It focuses specifically on running an already-trained network quickly and efficiently on a GPU for the purpose of generating a result (a process that is referred to in various places as scoring, detecting, regression, or inference).

Some training frameworks such as TensorFlow have integrated TensorRT so that it can be used to accelerate inference within the framework. Alternatively, TensorRT can be used as a library within a user application. It includes parsers for importing existing models from Caffe, ONNX, or TensorFlow, and C++ and Python APIs for building models programmatically.

<p align="center">
<img src = "https://github.com/SarthakGarg13/Severstal-Steel-Defect-Detection/blob/master/images/tensorrt.png">
</p>

## Model Conversion PyTorch-> ONNX-> TRT
TensorRT engine could be converted from the following frameworks using UFF parser, ONNX parser or TFTRT. The TensorRT API includes implementations for the most common deep learning layers. You can also use the C++ Plugin API or Python Plugin API to provide implementations for infrequently used or more innovative layers that are not supported out-of-the-box by TensorRT.

We have used the ONNX-Parser for the conversion Pytorch-TensorRT.
- Model is trained on Pytorch, and is saved as model.pth file. 
- Followed by converting .pth -> .onnx present in [simplify onnx model.ipynb](https://github.com/SarthakGarg13/Severstal-Steel-Defect-Detection/blob/master/Simplify%20onnx%20model.ipynb) 
- After converting model to onnx, we simplify it using the [Onnx-Simplifier](https://github.com/daquexian/onnx-simplifier).
- Followed by creating TensorRT engine(model.trt) and serializing it for later use.



<p align="center">
<img src = "https://github.com/SarthakGarg13/Severstal-Steel-Defect-Detection/blob/master/images/onnx-tensorrt.png">
</p>



## Environment

Replication of this repo is pretty simple.
- Install Docker
- Pull docker containers using:
  - docker pull "image-name:tag"
- Run container on Ubuntu using the command:
  - docker run -it -v "/path/to/directory" -p 1111:8888 "image-name"
- Open JupyterLab/JupyterNotebook
- Run the notebooks

### Docker Images

Docker Images for the Jupyter notebooks:
- Training Notebook: sg22/traindefect:version1
- Onnx-TRT Notebook: sg22/traindefect:version1
- Inference Notebook: sg22/defectinferencetrt:version1

## References

1. Mixed Precision Training Doc: https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html 
1. TensorRT Doc: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html
1. Onnx simplifier https://github.com/daquexian/onnx-simplifier
1. Dataset link https://www.kaggle.com/c/severstal-steel-defect-detection/data
1. U-Net: https://arxiv.org/abs/1505.04597
