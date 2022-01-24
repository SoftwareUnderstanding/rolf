English | [简体中文](README_CN.md)
# SegNet
SegNet
- [SegNet](#segnet)
  - [一、Introduction](#一introduction)
  - [二、Accuracy](#二accuracy)
  - [三、Dataset](#三dataset)
  - [四、Environment](#四environment)
  - [五、Quick Start](#五quick-start)
    - [Quick Train：](#quick-train)
    - [Quick Infer：](#quick-infer)
  - [六、Code structure and More explaination](#六code-structure-and-more-explaination)
  - [七、Model information](#七model-information)
## 一、Introduction

SegNet，a model for semantic segmentation  
Paper：<https://github.com/yassouali/pytorch-segmentation>  
Opensourced on AI Studio , can work online here：<https://aistudio.baidu.com/aistudio/projectdetail/2293857>

## 二、Accuracy
On the dataset camvid 11classses, miou = 0.601  

## 三、Dataset
Dataset used: camvid  
Download link：<https://aistudio.baidu.com/aistudio/datasetdetail/79232>  
This version come from：<https://www.kaggle.com/naureenmohammad/camvid-dataset>, in which resolution is low (480x360)，id of labels range from 0 to 11

Dataset Size：
- train_num：367
- val_num：101
- test_num：233

Format as follows：
- test           test_imgs（.png）
- testannot      test_labels（.png）
- train          train_imgs（.png）
- trainannot     train_labels（.png）
- val            val_imgs（.png）
- valannot       val_labels（.png）

## 四、Environment
- Hardware
  - GPU
  - CPU
- Framework / Software
  - PaddlePaddle = 2.1
  - PaddleSeg

## 五、Quick Start
### Quick Train：
```
cd PaddleSeg
python train.py \
       --config config.yml \
       --do_eval \
       --use_vdl \
       --save_interval 500 \
       --save_dir output
```
Parameters explaination：
- config ：root of config.yml
- do_eval ：validate while saving model
- use_vdl ：use VisualDl (a tool like tensorboard)
- save_interval ：frequency to save model
- save_dir ：directory to save models and logs

### Quick Infer：
```
python data/PaddleSeg/predict.py \
       --config config.yml \
       --model_path output_bs_8——pre/best_model/model.pdparams \
       --image_path data/PaddleSeg/camvid/test \
       --save_dir output/result
```
Parameters explaination：
- config ：root of config.yml
- model_path ：root of model to use
- image_path ：path of imgs needed to infer
- save_dir ：path to save outputs

## 六、Code structure and More explaination
This project is powered by PaddlePaddle ,code structrure is like PaddleSeg <https://github.com/PaddlePaddle/PaddleSeg>，to use, we just need to set the  `config.yml` file.

Introduction to  `config.yml` ：
```
batch_size: 4  #batch_size
iters: 1000    #iterations

train_dataset: #train_dataset config
  type: Dataset #type of train_dataset
  dataset_root: data/PaddleSeg/camvid #list of train_dataset
  train_path: data/PaddleSeg/camvid/train_list.txt #list of train_dataset
  num_classes: 12 #classed（background counts as well）
  transforms: #data pre-execution and enhancement
    - type: Resize #resize before put into network
      target_size: [512, 512] #resize to 512*512
    - type: RandomHorizontalFlip # use RandomHorizontalFlip to enhance data
    - type: Normalize #Normalize data
  mode: train

val_dataset: #train_dataset config
  type: Dataset #type of train_dataset
  dataset_root: data/PaddleSeg/camvid #list of train_dataset
  val_path: data/PaddleSeg/camvid/val_list.txt #list of train_dataset
  num_classes: 12 #classed（background counts as well）
  transforms: #data pre-execution and enhancement
    - type: Resize  #resize before put into network
      target_size: [512, 512]  #resize to 512*512
    - type: Normalize #Normalize img
  mode: val

optimizer: #set optimizer
  type: sgd #adopt SGD（Stochastic Gradient Descent）as optimizer
  momentum: 0.9 #set momentum
  weight_decay: 4.0e-5 #weight_decay, to avoid overfitting

learning_rate: #set learning_rate
  value: 0.1  #initial learning_rate
  decay:
    type: poly  #adopt poly as means of m
    power: 0.9  #rate of decay
    end_lr: 0   #end learning_rate

loss: #set loss funtion
  types:
    - type: CrossEntropyLoss #loss function type
  coef: [1]

model: #model config 
  type: SegNet  #set model category
  num_classes: 12
```

## 七、Model information
|  information   | description  |
|  ----  | ----  |
| Author  | chen jiejun |
| E-mail | stuartchen2018@outlook.com |
| Date  | 2021.8.8 |
|Framework version|PaddlePaddle = 2.1|
|Application scenarios| Semantic Segmantation|
|Support hardware|GPU CPU|
