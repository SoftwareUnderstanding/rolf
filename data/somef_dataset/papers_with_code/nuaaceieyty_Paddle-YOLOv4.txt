# Paddle-YOLOv4

English | [简体中文](./README_CN.md)

## 1、Introduction

This project is based on the paddlepaddle_V2.1 framework to reproduce YOLOv4. YOLOv4 is the fourth generation model of YOLO series. While retaining the encoding and decoding mode of YOLOv3 detection head, YOLOv4 uses stronger backbone network, stronger feature fusion module and more data enhancement modes, so that the model performance is significantly improved compared with YOLOv3, and the inference speed is still very fast.

**Paper:**
- [1] Bochkovskiy A, Wang C Y, Liao H. YOLOv4: Optimal Speed and Accuracy of Object Detection[J].2020.

**Reference project：**
- [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)

**The link of aistudio：**
- notebook：[https://aistudio.baidu.com/aistudio/projectdetail/2479219](https://aistudio.baidu.com/aistudio/projectdetail/2479219)

## 2、Accuracy

The model is trained on COCO2017's train set and tested on COCO2017's testdev set (the training can be verified on val set first to evaluate the training of the model).

The AP on testdev set given in the paper when the input size is 416x416 is 41.2%, and the AP obtained in this project is 41.2%.

![result](result.JPG)

## 3、Dataset

[COCO Dataset](https://aistudio.baidu.com/aistudio/datasetdetail/7122)
- Dataset size:
    - train set: 118,287
    - val set: 4952
    - testdev set: 20288
- Data format: standard COCO format, marked with rectangular boxes
## 4、Requirements

- Hardware：CPU、GPU（a machine with four Tesla V100-32G is recommended）

- Framework：
  - PaddlePaddle >= 2.1.2
  
## 5、Quick Start

### step1: clone 

```bash
# clone this repo
git clone https://github.com/nuaaceieyty/Paddle-YOLOv4.git
cd Paddle-YOLOv4
export PYTHONPATH=./
```
**Install dependencies**
```bash
pip install -r requestments.txt
```

### step2: Training

1. Create an Output folder in the top level directory and download CSPDarkNet backbone pre-training weights (I have converted the official CSPDarkNet weight to .pdparams) here: https://aistudio.baidu.com/aistudio/datasetdetail/103994.
2. This project can be trained using four card Tesla V100-32G. Note: COCO dataset should be prepared in advance, and decompressed into the data directory under the top-level directory (data set address is: https://aistudio.baidu.com/aistudio/datasetdetail/7122). If the dataset address is incorrect, change the corresponding address to the absolute path in the configs/datasets/coco_detection.yml file.

```bash
python -m paddle.distributed.launch --gpus 0,1,2,3 train.py -c configs/yolov4/yolov4_coco.yml --eval
```

### step3: Evaluating
Note: Make sure the best_model.pdparams file is in the output directory.
```bash
python3 eval.py -c configs/yolov4/yolov4_coco.yml
```

### step4: Generate results of testdev set for submission
Note: Make sure the best_model.pdparams file is in the output directory.
```bash
python3 eval.py -c configs/yolov4/yolov4_coco_test.yml
```
Zip the bbox. Json file generated in the live directory and send it to the evaluation server

### Prediction using pre training model

Put the images to be tested in the data directory, run the following command, and save the output images in the Output directory; If there is a GPU in the machine environment, delete -o use_gpu=False from the command

```bash
python3 predict.py -c configs/yolov4/yolov4_coco.yml --infer_img data/1.jpg -o use_gpu=False
```
The result is shown as follows：

![result](output/1.jpg)

## 六、Code structure

### 6.1 Structure

```
├─config                          
├─model                                                     
├─data                            
├─output                          
│  eval.py                        
│  predict.py                     
│  README.md                      
│  README_CN.md                   
│  requirements.txt               
│  train.py                       
```
### 6.2 Parameter description

Parameters related to training and evaluation can be set in `train.py`, as follows:

|  Parameters   | default  | description | other |
|  ----  |  ----  |  ----  |  ----  |
| config| None, Mandatory| Configuration file path ||
| --eval| None, Optional| Evaluate after an epoch |If you don't select this, you might have trouble finding the best_model|
| --fp16| None, Optional| Semi-precision training |If this option is not selected, 32GB of video memory may not be sufficient|
| --resume| None, Optional | Recovery training |For example: --resume output/yolov2_voc/66|

### 6.3 Training process

See 5、Quick Start

## 7、Model information

For other information about the model, please refer to the following table:

| information | description |
| --- | --- |
| Author | YU Tianyang(EICAS)|
| Date | 2021.10 |
| Framework version | Paddle 2.1.2 |
| Application scenarios | Object detection |
| Support hardware | GPU、CPU |
| Download link | [Pre training model](https://aistudio.baidu.com/aistudio/datasetdetail/107066)|
| Online operation | [notebook](https://aistudio.baidu.com/aistudio/projectdetail/2479219)|
