This repository contains Projectwork :arrow_down: undertaken for the partial fulfillment of Deep Learning Module and ETCS Creditpoints @OpenCampus.sh.

# Thoractic disease Detection on Chest X-ray Images

**Objective :** To train and explore Deep learning models for Medical Radiology Assistance ( Classification and Detection of  Thorax Diseases) using Chest X-ray datasets.

****

## :beginner: Index

1. Datasets

2. Implementation

3. References

****

## :diamond_shape_with_a_dot_inside: 1. Datasets

- [NIH Clinical Center Chest x-ray datasets y | National Institutes of Health (NIH)](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community)
  
  - :arrow_down: [Download Here](https://nihcc.app.box.com/v/ChestXray-NIHCC)
  - ⚠️ Incase of warning with Image folder !! :  run `python Utils/download.py`

## :computer: 2. Implementaition

**Status /Progress**

- [x]  Exploration of Datasets

- [x]  Exploartion of Application Domain and Challenges [Concepts: @AI for medical Diagnosis (coursera)]
  
  - [x]  [Class Imabalance Problem](https://github.com/Mnpr/OC-DeepLearning/blob/main/Documentation/ClassImbalance.ipynb)
  - [x]  Patient Overlap
  - [x]  [Explore Evaluation Metrics](https://github.com/Mnpr/OC-DeepLearning/blob/main/Documentation/EvaluationMetrics.ipynb)

- [x] Data Processing
  
  - [x] Data Preprocessing
  - [x] Image Processing
  - [x] Image Data Generator and Loaders

- [x]  [Mnist-D CNN Multiclass-Classifier](https://github.com/Mnpr/OC-DeepLearning/blob/main/CNN_Classifier_MnistD.ipynb)

- [x]  [`Multi-label Multi-Output` Classifier Subset  ](https://github.com/Mnpr/OC-DeepLearning/blob/main/CNN_Classifier_Subset.ipynb)

- [x]  [`Multi-label Multi-Output` Classifier Main](https://github.com/Mnpr/OC-DeepLearning/blob/main/CNN_Classifiers.ipynb)

- [x]  [Transfer Learning](https://github.com/Mnpr/OC-DeepLearning/blob/main/CNN_Classifiers.ipynb)
  
  - [x] Densenet-121

- [ ] Model Evaluation
  
  - [x]  Subset Dataset
  - [ ]  Full Dataset

- [x]  Local to Colab setup


## :bookmark_tabs: 3. References

- [1. ] [ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases](https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf)
- [2. ] [ CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays [arXiv:1711.05225v3  [cs.CV]  25 Dec 2017]](https://arxiv.org/pdf/1711.05225.pdf)
- [3.] [Densely Connected Convolutional Networks [arXiv:1608.06993v5 [cs.CV] 28 Jan 2018]](https://arxiv.org/pdf/1608.06993.pdf)
- [4.] [Coursera - AI for medical Diagnosis](https://www.coursera.org/learn/ai-for-medical-diagnosis/home/welcome)
- [5.] [Chest xray12 dataset problems : Single Radiologist Label Accuracy](https://lukeoakdenrayner.wordpress.com/2017/12/18/the-chestxray14-dataset-problems/)
