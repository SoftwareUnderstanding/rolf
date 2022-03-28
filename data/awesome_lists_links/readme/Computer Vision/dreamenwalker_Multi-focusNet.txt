# Multi-focusNet
This source code is used for the unpublished paper entitled “Multi-focus Network to Decode Imaging Phenotype for Overall Survival Prediction of Gastric Cancer Patients”. The code will be constantly updated based on comments from reviewers.

# Requirements
Our code is writen based on python 3.6,tensorflow 1.14.0, and keras 2.3.1.  We suggest that our code is run according to the requirements. Note that some errors may be occur during the training process using tensorflow 2.0.

# Description
Training.py：To train and evaluate the our proposed multi-focus network.

SurvmodelTMI2.py: The architecture of our proposed multi-focus network.

SurvmodelTMI_FPN18ori.py：The architecture of network based on the feature pyramid network and ResNet-18. More details will be found when our work is published.

SurvmodelTMI_FPN50ori.py The architecture of network based on the feature pyramid network and ResNet-50.

The directory of Methods contains the existing methods which are used for comparison.

# Acknowledgements
We are grateful to the authors for the published good work, in which the CheXNet(https://arxiv.org/pdf/1711.05225.pdf) is proposed. Our source codes with respect to the data augmentation and train evaluation are inspired by the CheXNet. The ResNet-18 and ResNet-50 can be found in the paper named “Deep Residual Learning for Image Recognition”. 
We also thank the work published in the paper named "Feature pyramid networks for object detection".
# Citation
If our code is helpful for you, please contact me.

# Contact
Please contact zhangliwen2018@ia.ac.cn if you have any questions.
