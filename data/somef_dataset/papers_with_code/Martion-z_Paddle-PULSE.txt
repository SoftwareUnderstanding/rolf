English | [简体中文](./README_cn.md) 

[TOC]

# 一、Introduction

An implementation of the SR models (`PULSE`) proposed in paper [PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models  ](https://arxiv.org/pdf/2003.03808v3.pdf) with PaddlePaddle.  PULSE traverses the high-resolution natural image manifold, searching for images that downscale to the original LR image.This is formalized through the “down-scaling loss,” which guides exploration through the latent space of a generative model.    

**Reference Project：**

\- [https://github.com/adamian98/pulse](https://github.com/adamian98/pulse)

**AI Studio Project：** 

\- notebook task：[https://aistudio.baidu.com/aistudio/projectdetail/2255411](https://aistudio.baidu.com/aistudio/projectdetail/2255411)

# 二、Accuracy

### 2.1 Visual Comparison

​    

![image](https://tva1.sinaimg.cn/large/008i3skNgy1gtcwqt3a1fj60ps0extah02.jpg)  

![image](https://tva1.sinaimg.cn/large/008i3skNgy1gtcwr0o65tj60pd0et75z02.jpg)  

### 2.2 NIQE 

**torch：average_NIQE=2.174**

![image](https://tva1.sinaimg.cn/large/008i3skNgy1gtcwbx6ua5j60q00lltc302.jpg)   

**paddle：average_NIQE=2.132**

![image](https://tva1.sinaimg.cn/large/008i3skNgy1gtcx4g6u1hj60q50nkwki02.jpg)

**pretrain GAN**

Address：[Baidu cloud disk](https://pan.baidu.com/s/1zRvbGmt7IOMoWSmQQz-ZHA)  Extraction code：f35u



# 三、Dataset

The input picture needs to be placed in the **input** folder, which is the same as [original reference code]（ https://github.com/adamian98/pulse ）Consistent, the input image is  16 × 16 celabahq face dataset:

[https://pan.baidu.com/s/1wGbZ4UxPDpQj2gV_Zq37pQ](https://pan.baidu.com/s/1wGbZ4UxPDpQj2gV_Zq37pQ)  Extraction code: mo0s

# 四、Environment

```
scipy==1.2.1
paddlepaddle==2.1.2
numpy==1.20.1
```

# 五、Quick Start

## Step1: Clone

```shell
# clone this repo
git https://github.com/Martion-z/Paddle-PULSE.git
cd Paddle-PULSE-main
```

## Step2: Load pretrain GAN model

Before running, you need to download the weight of the pre training network in advance.  Place the weight file(**styleGan.pdparams**) in the **models/cache** folder. The model address: 

[Baidu cloud disk](https://pan.baidu.com/s/1zRvbGmt7IOMoWSmQQz-ZHA)   Extraction code: f35u

## Step3: Run

```shell
python3 run.py
```

Just wait for the results. The output result (1024x1024) is stored in the **output1024** folder.

# 六、Code Structure and Explanation

## 6.1 Code Structure

```
./Paddle-Pulse
|-- images               
       |--input          #the path of input
       |--output1024		#the path of output
|-- models               
       |--cache 		#store the weight of model
       |--loss			#loss
       |--utils			#tool API
       |--pulse.py		#network of PULSE
       |--stylegan_paddle.py	#stylegan
|-- utils                #public tool API
|-- run.py					#main
|-- README.md            
|-- README_cn.md
```



## 6.2 Parameter Explanation

| Parameters | Default    | Explanation                              |
| ---------- | ---------- | ---------------------------------------- |
| input_dir  | input      | The path of the input                    |
| output_dir | Output1024 | The path of the output                   |
| batch_size | 1          | Batchsize                                |
| seed       | 0          | Random seed                              |
| eps        | 2e-3       | Optimizer                                |
| opt_name   | adam       | The class of Optimizer                   |
| steps      | 100        | The number of iterations to find the best picture |

# 七、Model Infomation

| Field                 | Content                 |
| :-------------------- | ----------------------- |
| Author                | 皮蛋瘦肉周                   |
| Date                  | 2021.08                 |
| Framework version     | paddlepaddle 2.1.2      |
| Application scenarios | Image Supper-Resolution |
| Supported hardware    | CPU、GPU                 |



