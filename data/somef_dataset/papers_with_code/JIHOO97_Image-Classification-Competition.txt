# Image Classification Competition
Competition의 목적은 사람 얼굴 이미지 데이터만을 가지고 마스크를 썼는지, 쓰지 않았는지, 정확히 쓴 것인지를 분류하는것에 있었습니다. 하지만 추가적으로 나이대, 성별 특성을 추가하여 총 18개의 분류로 나누는 것이 최종적인 목표가 되었습니다.

The goal of this competition is to classify an image of a person whether the person is wearing a mask, not wearing a mask, or incorrectly wearing a mask. However, age(<30, 30>=x<60, 60<=x) and gender(male, female) had to be additionally classified. Hence the final goal of this competition was to classify a total of 18 classes.

**Competition link:** [AI stage](https://stages.ai/)

## Hardware
- A server provided by UPStage
- GPU: V100

## Summary
1. [Installation](#installation)
2. [Data preprocessing](#data-preprocessing)
3. [3 models](#create-separate-models-for-age-mask-and-gender)
4. [1 model](#create-a-single-model-for-all-age-mask-and-gender)
5. [Voting](#voting)
6. [Grayscale images training](#grayscale-images-training)
7. [Cropped images training](#cropped-images-training)
8. [References](#references)

## Installation
Download all the required libraries with the following command.
```
pip install -r requirements.txt
```

## Dataset
데이터셋은 2700명의 사람들이 각각 마스크를 안 쓴 사진 1장, 쓴 사진 5장, 제대로 쓰지 않은 사진 1장으로 되어있습니다.
데이터는 공개 할 수 없습니다.

Each 2700 people took 7 images; 1 image not wearing a mask, 5 images wearing a mask, and 1 image incorrectly wearing a mask.
However, the dataset is private which is licensed by [UPStage](https://www.upstage.ai/).

## Data preprocessing
**File name**: label.ipynb

**label** file is to pre-process all the images. It has the following features.
  1. Display an image one by one using openCV, and press (1) to put it in the "wrong images" folder if the image is different from the description of the image
  2. Display an image in the "wrong images" folder one by one to check and ensure it is wrong
  3. Display an image in the filtered "wrong images" folder one by one, and type the corresponding class to correctly label it
  4. Combine the relabeled images to the original dataset

## Create separate models for age, mask, and gender
**File name:** three_models.ipynb

**model used:** resnet152
  1. Remove the last fully connected layer from the pretrained resnet152 model
  2. Add three different fully connected layers to the last layer of the model and apply them to age, mask, and gender respectively
  3. Generate the final result by summing the results from the three different fully connected layers

## Create a single model for all age, mask, and gender
**File name:** one_model.ipynb

Initialize wandb as the following.
```
wandb.init(project='your-project-name', entity='your-entity-name',config = {
    'learning_rate':0.001,
    'batch_size':16,
    'epoch':2,
    'model':'your-model-name',
    'momentum':0.9,
    'img_x':img_size_x[2],
    'img_y':img_size_y[2],
    'kfold_num':3,
})
config = wandb.config
```

Model | GPUs | Image size | Training Epochs | k-fold | batch size | learning_rate | momentum
------------ | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------
resnet152 | V100 | 224x224 | 2 | 3 | 16 | 0.001 | 0.9
vit_base_patch16_224 | V100 | 224x224 | 2 | 3 | 16 | 0.001 | 0.9
custom_model | V100 | 224x224 | 2 | 3 | 16 | 0.001 | 0.9

Model | Test Accuracy
------------ | -------------
vit_base_patch16_224 | 92.12
resnet152 | 91.64
custom_model | 2.51

Create a test dataset from "eval images" to evaluate the model

Model | Eval Accuracy (test) | Eval F1 score (test) | Eval Accuracy (final) | Eval F1 score (final)
------------ | ------------- | ------------- | ------------- | -------------
resnet152 | 80.460 | 0.774 | 79.937 | 0.755
vit_base_patch16_224 | 79.952 | 0.766 | 79.619 | 0.756

## Voting
**File name:** voting.ipynb

10 models with the best performance were softvoted to generate the final output

Combined Model (weight of resnet is **1**, weight of vit is **0.625**)
- resnet152
- resnet50
- resnet50 (complex transformation applied)
- resnet34
- resnet34 (mean and std for each image)
- vit_base_patch16_224(kfold5,epoch2, batch64)
- vit_base_patch16_224(stratified-kfold5, epoch1, cutmix-beta1, batch64)
- vit_base_patch16_224(kfold5, epoch1, batch64) 
- vit_base_patch16_224(stratified-kfold3, epoch5,  cutmix-beta1,batch64, swa)
- vit_large_patch16_224(stratified-kfold5, epoch1, batch 16)

Eval Accuracy (test) | Eval F1 score (test) | Eval Accuracy (final) | Eval F1 score (final)
------------ | ------------- | ------------- | -------------
81.635 | 0.781 | 81.000 | 0.771

## Grayscale images training
**File name:** gray_scale_train.ipynb

This file converts all the images into grayscale.
However, the pretrained cnn models only accept their input in rgb channels.
In order for the grayscale images to fit into the pretrained model, I had to change the input layer of the model.
After all, the result was not as good as expected.

## Cropped images training
**File name:** train_cropped.ipynb

This file crops only the face part of all the images.
I saved the cropped images separately in "cropped images" folder.
Resnet152 and different hyperparameters were tested using these images.
After all, the result of this model was not as good as expected.

## References
https://arxiv.org/pdf/1812.01187.pdf
https://kmhana.tistory.com/25
