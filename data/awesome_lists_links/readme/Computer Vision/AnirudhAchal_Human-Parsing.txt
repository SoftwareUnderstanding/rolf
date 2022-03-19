# Human Parsing IEEE
This project was taken as a year long project under IEEE NITK 2020.

## Demo

![](Demo/Human-Parsing-Demo-1.gif)

![](Demo/Human-Parsing-Demo-2.gif)

## Model
The implementation of PSPNet is based on [here](https://github.com/Lextal/pspnet-pytorch).


## Environment
All the required libraries can be found in requirements.txt. They are also listed below.
```
click==7.1.2
cycler==0.10.0
Flask==1.1.2
imageio==2.9.0
itsdangerous==1.1.0
Jinja2==2.11.3
kiwisolver==1.3.1
MarkupSafe==1.1.1
matplotlib==3.3.3
numpy==1.19.5
Pillow==8.1.0
pyparsing==2.4.7
python-dateutil==2.8.1
six==1.15.0
torch==1.7.1
torchvision==0.8.2
typing-extensions==3.7.4.3
Werkzeug==1.0.1

```

## Project Structure
```
Human-Parsing
├── app.py
├── checkpoints
│   ├── densenet
│   ├── resnet101
│   ├── resnet121
│   ├── resnet18
│   ├── resnet34
│   └── resnet50
├── Datasets
│   └── lip.py
├── eval.py
├── inference.py
├── Net
│   ├── extractors.py
│   └── pspnet.py
├── README.md
├── requirements.txt
├── static
│   ├── bg_image.jpeg
│   ├── input.png
│   └── output.png
├── templates
│   ├── display.html
│   └── home.html
└── train.py
```

## Usage

```
python3 train.py -d [DATAPATH] -e [EPOCHS] -b [BATCHSIZE] --backend [densenet|resnet50|resnet34] 

python3 eval.py -d [DATAPATH] -b [BATCHSIZE] --backend [densenet|resnet50|resnet34] --visualize

python3 inference.py -d [IMAGE_DATAPATH] --backend [densenet|resnet50|resnet34] 
```
For each of these files, to view all the options available during training and evaluation, use --help or -h as shown below.

```
python3 train.py --help
python3 eval.py --help
python3 inference.py --help
```

## Dataset
The dataset can be downloaded from [here](https://drive.google.com/drive/folders/1ZjNrTb7T_SsOdck76qDcd5OHkFEU0C6Q).
The structure of the dataset is shown below.
```
LIP
├── Testing_images
│   ├── test_id.txt
│   └── testing_images [10000 entries]
├── train_segmentations_reversed [30462 entries]
├── TrainVal_images
│   ├── train_id.txt
│   ├── train_images [30462 entries]
│   ├── val_id.txt
│   └── val_images [10000 entries]
├── TrainVal_parsing_annotations
│   ├── README_parsing.md
│   ├── train_segmentations [30462 entries]
│   └── val_segmentations [10000 entries]
└── TrainVal_pose_annotations
    ├── lip_train_set.csv
    ├── lip_val_set.csv
    ├── README.md
    └── vis_annotation.py

```

## References
- https://github.com/Lextal/pspnet-pytorch
- https://github.com/hyk1996/Single-Human-Parsing-LIP
- https://arxiv.org/abs/1612.01105
