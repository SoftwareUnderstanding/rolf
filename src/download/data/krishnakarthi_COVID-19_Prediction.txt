# COVID-19 Prediction using Deep Transfer Learning in Azure ML Workspace

### Abstract : 
Diagnose the COVID-19 from patients chest X-ray's  using Convolution Neural Networks (CNN) Deep Transfer Learning technique in Azure ML workspace.

### Dataset Information:
Both COVID-19 and Normal patient chest X-ray dataset collection from open source public dataset.

For COVID-19 positive patients dataset has been collected from below GitHub repository, it has chest X-ray & CT scan image of patients which are positive or suspected of COVID-19 or other viral and bacterial pneumonias (MERS, SARS, and ARDS.). 
From this dataset we have collected only COVID-19 patient’s chest X-ray images and it has around 235 images which we are used for our prediction.

**COVID-19 Positive patient's chest X-ray Data :**  https://github.com/ieee8023/covid-chestxray-dataset 

For NORMAL patient’s chest X-rays dataset has been collected from Kaggle. It has Normal and Pneumonia patient chest X-rays data.From that we have collected only healthy patiet's 231 chest X-rays image to make sure dataset are balanced with COVID-19 data.

**Healthy patient's chest X-ray Data :** https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

![Alt text](https://github.com/krishnakarthi/COVID-19_Prediction/blob/master/Image/total%20Class.JPG?raw=true "Title")

### Datasplit (Train, Validation and Test) :

1. Total Data Count       : 466
2. Train Data Count       : 298
3. Validation Data Count  : 75
4. Test Data Count        : 93

![Alt text](https://github.com/krishnakarthi/COVID-19_Prediction/blob/master/Image/Data_Split.JPG?raw=true "Title")

### Image Data Pre-Processing :

For image data pre-processing we  have done below given steps in train,validation and test dataset. 

 - We have used Open CV package to resize image data to 150x150.
 - Normalize the image data to "0 to 1" from "0 to 255".
 

### Deep Transfer Learning Model:

After pre-processed data are re-trained with below given image recognition models using Deep Transfer Learning technique.
From model evaluation we are able to see Xception and InceptionV3 model giving promising result for this dataset.

1. Xception     : https://arxiv.org/abs/1610.02357
2. InceptionV3  : https://arxiv.org/abs/1512.03385
3. DenseNet201  : https://arxiv.org/abs/1608.06993
4. ResNet50     : https://arxiv.org/abs/1512.03385

**Find models evaluation with test data  :**

![Alt text](https://github.com/krishnakarthi/COVID-19_Prediction/blob/master/Image/Final_Result.JPG?raw=true "Title")


**Test dataset Confusion Matrix for Xception Model :**

![Alt text](https://github.com/krishnakarthi/COVID-19_Prediction/blob/master/Image/Xception_Confusion%20Matrix.JPG?raw=true "Title")

**Test dataset Confusion Matrix for InceptionV3 Model :**

![Alt text](https://github.com/krishnakarthi/COVID-19_Prediction/blob/master/Image/Inception_ConfusionMatrix.JPG?raw=true "Title")


### Azure ML Workspace:

From trained models, we have selected Xception model which has best result for this dataset and deployed into Azure ML.

**Note : Please make sure you have first craeted Azure ML workspace. This script will help to register custom mode into Azure ML workspace and deploy into ACI (Azure Container Instances).**

Refer below notebook to deploy custom built model into Azure ML which has given in this repository.

https://github.com/krishnakarthi/COVID-19_Prediction/blob/master/DeployTrainedModelToAzureML.ipynb


### Requirements:
 - Tensorflow == 2.2.0
 - Keras == 2.3.1
 - Python == 3.7.3
 - Azure ML SDK Version == 1.4.0
 

### Reference:

https://github.com/ieee8023/covid-chestxray-dataset

https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-existing-model

https://docs.microsoft.com/en-us/azure/machine-learning/how-to-consume-web-service

https://docs.microsoft.com/en-us/azure/azure-functions/functions-develop-vs

https://keras.io/applications/



### Author:
*Karthikeyan Krishnasamy*

https://www.linkedin.com/in/karthikeyankrishnasamy/ 
