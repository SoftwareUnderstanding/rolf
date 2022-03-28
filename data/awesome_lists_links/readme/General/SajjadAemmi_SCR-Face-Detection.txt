## Introduction

SCRFD is an efficient high accuracy face detection approach which initially described in [Arxiv](https://arxiv.org/abs/2105.04714).

<img src="https://github.com/nttstar/insightface-resources/blob/master/images/scrfd_evelope.jpg" width="400" alt="prcurve"/>

## Performance

Precision, flops and infer time are all evaluated on **VGA resolution**.

#### ResNet family

| Method              | Backbone        | Easy  | Medium | Hard  | \#Params(M) | \#Flops(G) | Infer(ms) |
| ------------------- | --------------- | ----- | ------ | ----- | ----------- | ---------- | --------- |
| DSFD (CVPR19)       | ResNet152       | 94.29 | 91.47  | 71.39 | 120.06      | 259.55     | 55.6      |
| RetinaFace (CVPR20) | ResNet50        | 94.92 | 91.90  | 64.17 | 29.50       | 37.59      | 21.7      |
| HAMBox (CVPR20)     | ResNet50        | 95.27 | 93.76  | 76.75 | 30.24       | 43.28      | 25.9      |
| TinaFace (Arxiv20)  | ResNet50        | 95.61 | 94.25  | 81.43 | 37.98       | 172.95     | 38.9      |
| - | - | - | - | - | - | - | - |
| ResNet-34GF         | ResNet50        | 95.64 | 94.22  | 84.02 | 24.81       | 34.16      | 11.8      |
| **SCRFD-34GF**      | Bottleneck Res  | 96.06 | 94.92  | 85.29 | 9.80        | 34.13      | 11.7      |
| ResNet-10GF         | ResNet34x0.5    | 94.69 | 92.90  | 80.42 | 6.85        | 10.18      | 6.3       |
| **SCRFD-10GF**      | Basic Res       | 95.16 | 93.87  | 83.05 | 3.86        | 9.98       | 4.9       |
| ResNet-2.5GF        | ResNet34x0.25   | 93.21 | 91.11  | 74.47 | 1.62        | 2.57       | 5.4       |
| **SCRFD-2.5GF**     | Basic Res       | 93.78 | 92.16  | 77.87 | 0.67        | 2.53       | 4.2       |


#### Mobile family

| Method              | Backbone        | Easy  | Medium | Hard  | \#Params(M) | \#Flops(G) | Infer(ms) |
| ------------------- | --------------- | ----- | ------ | ----- | ----------- | ---------- | --------- |
| RetinaFace (CVPR20) | MobileNet0.25   | 87.78 | 81.16  | 47.32 | 0.44        | 0.802      | 7.9       |
| FaceBoxes (IJCB17)  | -               | 76.17 | 57.17  | 24.18 | 1.01        | 0.275      | 2.5       |
| - | - | - | - | - | - | - | - |
| MobileNet-0.5GF     | MobileNetx0.25  | 90.38 | 87.05  | 66.68 | 0.37        | 0.507      | 3.7       |
| **SCRFD-0.5GF**     | Depth-wise Conv | 90.57 | 88.12  | 68.51 | 0.57        | 0.508      | 3.6       |


**X64 CPU Performance of SCRFD-0.5GF:**

| Test-Input-Size         | CPU Single-Thread   | Easy  | Medium | Hard  |
| ----------------------- | -----------------   | ----- | ------ | ----- |
| Original-Size(scale1.0) | -                   | 90.91 | 89.49  | 82.03 |
| 640x480                 | 28.3ms              | 90.57 | 88.12  | 68.51 |
| 320x240                 | 11.4ms              | -     | -      | -     |

*precision and infer time are evaluated on AMD Ryzen 9 3950X, using the simple PyTorch CPU inference by setting `OMP_NUM_THREADS=1` (no mkldnn).*

## Installation

Please refer to [mmdetection](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md) for installation.
 
  1. Install [mmcv](https://github.com/open-mmlab/mmcv). (mmcv-full==1.2.6 and 1.3.3 was tested)
  2. Install build requirements and then install mmdet.
       ```
       pip install -r requirements/build.txt
       pip install -v -e .  # or "python setup.py develop"
       ```

## Pretrained-Models

|      Name      | Easy  | Medium | Hard  | FLOPs | Params(M) | Infer(ms) | Link                                                         |
| :------------: | ----- | ------ | ----- | ----- | --------- | --------- | ------------------------------------------------------------ |
|   SCRFD_500M   | 90.57 | 88.12  | 68.51 | 500M  | 0.57      | 3.6       | [download](https://1drv.ms/u/s!AswpsDO2toNKqyYWxScdiTITY4TQ?e=DjXof9) |
|    SCRFD_1G    | 92.38 | 90.57  | 74.80 | 1G    | 0.64      | 4.1       | [download](https://1drv.ms/u/s!AswpsDO2toNKqyPVLI44ahNBsOMR?e=esPrBL) |
|   SCRFD_2.5G   | 93.78 | 92.16  | 77.87 | 2.5G  | 0.67      | 4.2       | [download](https://1drv.ms/u/s!AswpsDO2toNKqyTIXnzB1ujPq4th?e=5t1VNv) |
|   SCRFD_10G    | 95.16 | 93.87  | 83.05 | 10G   | 3.86      | 4.9       | [download](https://1drv.ms/u/s!AswpsDO2toNKqyUKwTiwXv2kaa8o?e=umfepO) |
|   SCRFD_34G    | 96.06 | 94.92  | 85.29 | 34G   | 9.80      | 11.7      | [download](https://1drv.ms/u/s!AswpsDO2toNKqyKZwFebVlmlOvzz?e=V2rqUy) |
| SCRFD_500M_KPS | 90.97 | 88.44  | 69.49 | 500M  | 0.57      | 3.6      | [download](https://1drv.ms/u/s!AswpsDO2toNKri_NDM0GIkPpkE2f?e=JkebJo) |
| SCRFD_2.5G_KPS | 93.80 | 92.02  | 77.13 | 2.5G  | 0.82      | 4.3       | [download](https://1drv.ms/u/s!AswpsDO2toNKqyGlhxnCg3smyQqX?e=A6Hufm) |
| SCRFD_10G_KPS  | 95.40 | 94.01  | 82.80 | 10G   | 4.23      | 5.0       | [download](https://1drv.ms/u/s!AswpsDO2toNKqycsF19UbaCWaLWx?e=F6i5Vm) |

mAP, FLOPs and inference latency are all evaluated on VGA resolution.
``_KPS`` means the model includes 5 keypoints prediction.

## Convert to ONNX

Please refer to `tools/scrfd2onnx.py`

Generated onnx model can accept dynamic input as default.

You can also set specific input shape by pass ``--shape 640 640``, then output onnx model can be optimized by onnx-simplifier.


## Inference
Put your input images or videos in `./input` directory. The output will be saved in `./output` directory. 
In root directory of project, run the following command for image: 

```
python inference_image.py --input "./input/test.jpg"
```
and for video:
```
python inference_video.py --input "./input/obama.mp4"
```
Use -sh for show results during code running or not

Note that you can pass some other arguments. Take a look at `inference_video.py` file.
