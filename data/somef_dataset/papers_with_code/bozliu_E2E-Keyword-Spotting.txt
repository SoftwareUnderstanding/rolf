# E2E-Keyword-Spotting

Joint End to End Approaches to Improving Far-field Wake-up Keyword Detection

## :wrench: Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.3](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)


1. Install dependent packages

    ```bash
    cd E2E-Keyword-Spotting
    pip install -r requirements.txt
    ```
2. Or use conda 
    ```bash
    cd E2E-Keyword-Spotting
    conda env create -f environment.yaml
    ```

## :turtle: Dataset Preparation

#### How to Use
Dataset is from [Google Speech Command](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) published in  [arxiv](https://arxiv.org/abs/1804.03209).
* Data Pre-processing (Has already been done)
1. According to the file, dataset has already been splited into three folders, train, test, and valid. 
1. The splited [Google Speech Command dataset](https://drive.google.com/file/d/1InqR8n7l5Qj6voJREpcjHYWHVTKG-BbB/view?usp=sharing) is saved in Google Drive folder. 
    
## :computer: Train and Test
### Training commands
- **Single GPU Training**: 
```
python train.py
```
- **Distributed Training**: 
```
CUDA_VISIBLE_DEVICES=0,1 python train.py
```
### Test commands
```
python test.py 
```
## Neural Network Architectures
### General Architecture
<img src="https://github.com/bozliu/E2E-Keyword-Spotting/blob/main/images/Standard%20E2E%20Architecture.png " width="100%">

1. Multi-head Attention
<img src="https://github.com/bozliu/E2E-Keyword-Spotting/blob/main/images/Multi-Head%20Attention%20Architecture.png" width="100%">
* Encoder: GRU/LSM 
* Attention Heads: 8
* GRU hidden nodes: 128/256/512
* GRU layers: 1/2/3
* Increasing GRU hidden layers nodes can increase the performance much better than increasing hidden layers 

2. VGG19/VGG16/VGG13/VGG11 with/without batch normalization
3. Deep Residual Neural Network ('resnet18', 'resnet34', 'resnet50') 
4. Wide Residual Networks ('wideresnet28_10') imported from the [repository](https://github.com/xternalz/WideResNet-pytorch/blob/master/wideresnet.py)
5. Dual Path Networks from [arxiv](https://arxiv.org/abs/1707.01629) 
6. Densely Connected Convolutional Networks from [arxiv](https://arxiv.org/abs/1608.06993)

## Result

### Model Parameters 
<img src="images/model_parameters.png" width="100%">

### Best Accuracy Training Process
![image](https://github.com/bozliu/E2E-Keyword-Spotting/blob/main/images/best%20accuracy.png)

### Best Loss Training Process
![image](https://github.com/bozliu/E2E-Keyword-Spotting/blob/main/images/best%20loss.png)

## Files Description  
├── kws   
│   ├── metrics    
│   │   ├── fnr_fpr.py  
│   │   ├── __init__.py  
│   ├── models   
│   │   ├── attention.py  
│   │   ├── crnn.py  
│   │   ├── densenet.py  
│   │   ├── dpn.py  
│   │   ├── __init__.py  
│   │   ├── resnet.py   
│   │   ├── resnext.py  
│   │   ├── treasure_net.py  
│   │   ├── vgg.py  
│   │   └── wideresnet.py  
│   ├── transforms  
│   ├── utils.py  
├── config.py  

* *./kws/metrics* : Evaluation matrics, defining the False Rejection Rate (FRR) and False Alarm Rate (FAR) for keyword spotting
* *./kws/models* : Diffferent network architecture 
* *.config.py* : Configuration about parameters and hyperparameters
