## CheXNet Exploration

### Source of the ChestX-ray14 paper and data
 - __original paper__: [https://arxiv.org/pdf/1705.02315.pdf](https://arxiv.org/pdf/1705.02315.pdf)
 - __paper by NG__: [https://arxiv.org/abs/1711.05225](https://arxiv.org/abs/1711.05225)
 - __dataset_nih__: [https://nihcc.app.box.com/v/ChestXray-NIHCC](https://nihcc.app.box.com/v/ChestXray-NIHCC)
 - __dataset_kaggle__: [https://www.kaggle.com/nih-chest-xrays/data](https://www.kaggle.com/nih-chest-xrays/data)


### Analysis of dataset
 - __concerns from a radiologist__: [Exploring the ChestXray14 dataset: problems](https://lukeoakdenrayner.wordpress.com/2017/12/18/the-chestxray14-dataset-problems/)

### Model implementation
Since the models described in source paper and paper by NG's group are not available, I did some search and located some useful dataset analyze and model exploration efforts made by people on Kaggle and Github.
 - _Kevin Mader_ on Kaggle. He has presented very good dataset analysis, data preprocessings and [model training](https://www.kaggle.com/kmader/cardiomegaly-pretrained-vgg16/notebook). 
 - _Caleb P_ on Kaggle. Tried to MobileNet and InceptionResNetV2 as base model and showed sample [data training and result presenting](https://www.kaggle.com/cpagel/adjust-simple-xray-cnn/notebook).
 - _arnoweng_ on GitHub. It is a [pytorch reimplementation](https://github.com/arnoweng/CheXNet) of CheXNet, that presented by NG paper. It only has prediction code, not training code.
 - _brucechou1983_ on GitHub. It is a [keras reimplementation](https://github.com/brucechou1983/CheXNet-Keras) of CheXNet. Contains full codes. 


### My exploration
 - _cnn1, cnn2_: Used the data preprocessing and model building method described by Kevin Mader.
 - _cnn3_: Used data preprocessing method by Kevin Mader and model building method by Caleb P.
 - _cnn4_: Used data preprocessing method by Kevin Mader and model building method by brucechou1983. Updated image proprocessing method: center cropping, 0-1 normalization, mean/std normalization. Train on 2 classes.
 - _cnn5_: Used data preprocessing method by Kevin Mader and model building method by brucechou1983. Updated image proprocessing method: center cropping, 0-1 normalization, mean/std normalization. Train on 14 classes.
 - _cnn6_: Used data preprocessing method and model building method by brucechou1983. Followed the styles of optimizer, learningratescheduler, class_weights. Train on 14 classes.
 - _cnn7_: Used data preprocessing method and model building method by brucechou1983. Followed the styles of optimizer, learningratescheduler, class_weights. Train on 2 classes.
 - _cnn8_: Used InceptionV3 as base model, train on 2 classes.


### Results
 - per-class AUROC value of model trained with bruce's code, in CheXNet-Keras/experiments/1.
 
|     Pathology      | [Wang et al.](https://arxiv.org/abs/1705.02315) | [Yao et al.](https://arxiv.org/abs/1710.10501) | [CheXNet](https://arxiv.org/abs/1711.05225) | Our Model |
| :----------------: | :-------------: | :-------------: | :--------------: | :--------------: |
|    Atelectasis     |      0.716      |      0.772      |      0.8094      |      0.7810      |
|    Cardiomegaly    |      0.807      |      0.904      |      0.9248      |      0.8694      |
|      Effusion      |      0.784      |      0.859      |      0.8638      |      0.8690      |
|    Infiltration    |      0.609      |      0.695      |      0.7345      |      0.7066      |
|        Mass        |      0.706      |      0.792      |      0.8676      |      0.7322      |
|       Nodule       |      0.671      |      0.717      |      0.7802      |      0.6123      |
|     Pneumonia      |      0.633      |      0.713      |      0.7680      |      0.6867      |
|    Pneumothorax    |      0.806      |      0.841      |      0.8887      |      0.8224      |
|   Consolidation    |      0.708      |      0.788      |      0.7901      |      0.7242      |
|       Edema        |      0.835      |      0.882      |      0.8878      |      0.8918      |
|     Emphysema      |      0.815      |      0.829      |      0.9371      |      0.8743      |
|      Fibrosis      |      0.769      |      0.767      |      0.8047      |      0.7158      |
| Pleural Thickening |      0.708      |      0.765      |      0.8062      |      0.7725      |
|       Hernia       |      0.767      |      0.914      |      0.9164      |      0.8020      |


#### Useful links
 - keras checkpoint saving: https://machinelearningmastery.com/check-point-deep-learning-models-keras/