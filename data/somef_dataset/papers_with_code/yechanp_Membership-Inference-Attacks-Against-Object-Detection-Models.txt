# Membership-Inference-Attacks-Against-Object-Detection-Models

This is the codes for  https://arxiv.org/abs/2001.04011 (Membership Inference Attacks Against Object Detection Models).


## Requirements

* Python
* Chainer 
* Pytorch

## Training Shadow and Target Models
To train object detection models, please run the following files.

```
# For SSD300 using VOC dataset
python train_Chainer_ssd_voc_shadow.py  [gpu_id]
python train_Chainer_ssd_voc_target.py  [gpu_id]

```

To train the attacker using the shadow model with the shadow dataset, please execute 
```
python mia_train_attacker_.py  config.py
```
To attack the target object detection model, please execute
```
python mia_evaluate.py  config.py
```

## Trained Object Detection Model Weights 

For VOC-SSD300 models, you can download the weights of the shadow and target models at https://drive.google.com/drive/folders/1E45q9K1kqLzVqyYrrpZid07oq4JN5Z-V?usp=sharing .

Futher model weights will be uploaded later.
