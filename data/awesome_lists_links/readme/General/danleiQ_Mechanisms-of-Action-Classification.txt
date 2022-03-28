# Mechanisms of Action Classification


This repo is intended to share the solutions of the Kaggle Competition- Mechanisms of Action (MoA) Prediction. The goal of the competition is to “classify drugs based on their biological activity”. This is a multi-label classification task. Drugs can have multiple MoA annotations which describe binary responses from different cell types in different ways. 

You can download datasets of the competition from the following link:
https://www.kaggle.com/c/lish-moa/overview

# Installation
#### Tabnet -Pytorch
- [![CircleCI](https://circleci.com/gh/dreamquark-ai/tabnet.svg?style=svg)](https://circleci.com/gh/dreamquark-ai/tabnet)

- [![PyPI version](https://badge.fury.io/py/pytorch-tabnet.svg)](https://badge.fury.io/py/pytorch-tabnet)

- ![PyPI - Downloads](https://img.shields.io/pypi/dm/pytorch-tabnet)

- or you can install using pip by running: pip install pytorch-tabnet
#### 2-Phase NN with Transfer Learning- Pytorch

#### 2-Head ResNet-like NN- Tensorflow
# Model Architecture
![image](https://github.com/danleiQ/Mechanisms-of-Action-Classification/blob/master/Presentation/model%20diagram.jpg)

# Result
| Single Model | Seeds | K-folds | Cross Validation without Drug_id | Cross Validation with Drug_id | Public Score | Private Score | 
| ----- | ----- | ----- | ----- | ----- | ----- |  ----- | 
| Tabnet | 1 |10 | 0.016717 |  |0.01841| 0.01632 |
| 2-Phase NN With Transfer Learning | 7 | 7 | |  0.01563 |0.01833| 0.01623 |
|2-Heads Resnet NN | 7 |10 |0.01656 |   |0.01850| 0.01635 |
|Ensemble with average weights |  | | |   |0.01824| 0.01609 |

# References
Here are some resources I've been learning from:

### Model Achitecture:

#### TabNet:

- Paper: https://arxiv.org/abs/1908.07442

- Pytorch Implementation: https://github.com/dreamquark-ai/tabnet

- Tensorflow 2.0 Implementation: https://www.kaggle.com/marcusgawronsky/tabnet-in-tensorflow-2-0
 
- Public Notebook:https://www.kaggle.com/marcusgawronsky/tabnet-in-tensorflow-2-0

#### Transfer Learning:

- kaggle notebook:https://www.kaggle.com/chriscc/kubi-pytorch-moa-transfer
               - https://www.kaggle.com/thehemen/pytorch-transfer-learning-with-k-folds-by-drug-ids
                 
                 
#### ResNet-like NN
- https://www.kaggle.com/demetrypascal/2heads-deep-resnets-pipeline-smoothing
- https://www.kaggle.com/rahulsd91/moa-multi-input-resnet-model

-----------------
### Feature Importance:

#### Permutation Importance:

- Introduction:https://www.kaggle.com/dansbecker/permutation-importance

- eli5 implementation: https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html

- sklearn implementation: https://scikit-learn.org/stable/modules/permutation_importance.html

#### T-test

- https://www.kaggle.com/demetrypascal/t-test-pca-rfe-logistic-regression#Select-only-important

#### Adversarial Validation

- https://towardsdatascience.com/adversarial-validation-ca69303543cd

### Feature Engineering:

#### K-means:
- https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1

- https://www.kaggle.com/yerramvarun/deciding-clusters-in-kmeans-silhouette-coeff

- https://towardsdatascience.com/silhouette-coefficient-validating-clustering-techniques-e976bb81d10c

#### PCA 
- https://www.mikulskibartosz.name/pca-how-to-choose-the-number-of-components/

- https://www.kaggle.com/kushal1506/deciding-n-components-in-pca


-----------------
### Hyperparameters Tuning:

Optuna: 

- Tutorial: https://www.kaggle.com/fanvacoolt/tutorial-on-hyperopt

- Optuna: https://optuna.readthedocs.io/en/v2.1.0/reference/generated/optuna.visualization.plot_intermediate_values.html

-----------------
### Label Smoothing

https://leimao.github.io/blog/Label-Smoothing

