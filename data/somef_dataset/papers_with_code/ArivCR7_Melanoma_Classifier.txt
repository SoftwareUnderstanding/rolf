# Melanoma_Classifier

# Melanoma - Introduction
Skin cancer is the most prevalent type of cancer. Melanoma, specifically, is responsible for 75% of skin cancer deaths, despite being the least common skin cancer. As with other cancers, early and accurate detection—potentially aided by data science—can make treatment more effective.

Currently, dermatologists evaluate every one of a patient's moles to identify outlier lesions or “ugly ducklings” that are most likely to be melanoma. Existing AI approaches have not adequately considered this clinical frame of reference. Dermatologists could enhance their diagnostic accuracy if detection algorithms take into account “contextual” images within the same patient to determine which images represent a melanoma. If successful, classifiers would be more accurate and could better support dermatological clinic work.

Melanoma is a deadly disease, but if caught early, most melanomas can be cured with minor surgery. Image analysis tools that automate the diagnosis of melanoma will improve dermatologists' diagnostic accuracy. Better detection of melanoma has the opportunity to positively impact millions of people.

# Normal Skin Lesion
<img src="static/competitions_20270_1222630_jpeg_test_ISIC_0052349.jpg" alt="Normal Skin lesion" width=200 height=200/>            

# Melanoma Skin Lesion                                                                                                      
<img src="static/c5.jpg" alt="Melanoma" width=200 height=200/>


# About Data and Code

This is an ongoing kaggle contest hosted by Society for Imaging Informatics in Medicine (SIIM). SIIM have provided 100GB data of melanoma skins out of which 98.3% data belong to normal skin(negative class) and just the remaining 1.7% belong to melanoma skin lesions(positive class). There is a huge class imbalance here and I've addressed it in my code by giving weightage to positive class predictions. 
Use the code melanoma-pytorch-starter to train your own model.

Since the data is huge in size(100GB), I used the kaggle in-house dataset which is the numpy converted data of the image. So to run the above notebook in local, we need the following kaggle datasources: 'SIIM-ISIC Melanoma Resized Images' and 'Pretrained Model Weights(Pytorch)'. There are multiple resolutions of resized images in the above mentioned datasource of which I used 128x128 resolution. The higher the resolution we choose, the higher will be the model performance as will be more pixel variance the model can learn. But there comes a trade off for memory(RAM) limitation. The higher the resolution, higher will be the memory required to store the image, that leads to run-time out of memory error. 

The CNN architecture used here is SEResNext50_32x4d after some random trials with resnet18 and resnet50. More about this architecture on https://arxiv.org/pdf/1709.01507.pdf

# Training - Evaluation

Data is split into 5 different folds and the model got trained on all the folds individually and saved. 
Learning Rate scheduler is used to decay the learning rate after definite set of epochs.
Early Stopping is enabled to avoid overfitting the model.
Since it's a highly imbalanced dataset, accuracy alone will not give a good picture of the model. The metric used here is AUROC which takes values between 0-1. The closer the value of AUROC to 1, the better will be the model. The AUROC of our model is 0.89.

# Exposed model as a service

I've created an API for this model, where we can pass an image of our skin lesion and check the probability of melanoma on that lesion. 

<img src="static/2020-07-13-08-38-05.gif" alt="Melanoma" width=500 height=300/>

