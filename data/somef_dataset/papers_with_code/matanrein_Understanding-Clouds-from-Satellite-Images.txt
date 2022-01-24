# Understanding-Clouds-from-Satellite-Images

This is a solution to the "Understanding Clouds From Satellite Images" Kaggle Competition(https://www.kaggle.com/c/understanding_cloud_organization).
In this competition we were required to detect clouds in images and classify them into 4 different cloud types(more information in the competition link above).
I was supplied with a dataset of ~5000 images annotated by scientists for training. 
The Prediction input is a sattelite image. The output is a prediction, for each pixel and each cloud type, whether this type appears in the pixel.

To solve this problem of Semantic Segmentation, I used the Unet CNN architechture(https://arxiv.org/abs/1505.04597) with efficientnet backbone.

## Instructions to run my project

1. clone the project from github: `git clone https://github.com/matanrein/Understanding-Clouds-from-Satellite-Images.git`
2. `pip install -r requirements.txt`
3. download competition files: `kaggle competitions download understanding_cloud_organization`
4. unzip competition files: `unzip  understanding_cloud_organization.zip`
5. generate resized images:  `PYTHONPATH=Understanding-Clouds-from-Satellite-Images/ python Understanding-Clouds-from-Satellite-Images/com/understandingclouds/preprocessing/resize_images.py`
6. generate augmented images: `PYTHONPATH=Understanding-Clouds-from-Satellite-Images/ python Understanding-Clouds-from-Satellite-Images/com/understandingclouds/preprocessing/data_augmentation.py`
7. run training: `PYTHONPATH=Understanding-Clouds-from-Satellite-Images/ python Understanding-Clouds-from-Satellite-Images/com/understandingclouds/training.py`
8. predict for test set:  `PYTHONPATH=Understanding-Clouds-from-Satellite-Images/ python Understanding-Clouds-from-Satellite-Images/com/understandingclouds/predict.py`
9. submit results:  `kaggle competitions submit -f submission.csv -m "test submission" understanding_cloud_organization`
