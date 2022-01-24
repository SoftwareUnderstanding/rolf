# TransferLearning_DiscriminativeFineTuning

The purpose of this repository is to illustrate the implementation of Discriminative Fine Tuning for Transfer learning models in Keras. The key idea of Discriminative Fine Tuning (let's call it DFT going foward) is explained by Jeremy Howard and Sebastian Ruder in their 2018 paper "Universal Language Model Fine-tuning for Text Classification" https://arxiv.org/pdf/1801.06146.pdf

The fast.ai library comes inbuilt with DFT. I found two implemetations of DFT for Keras in public domain - one for SGD and the other one for Adam. The links for these two are given below: 1) SGD - https://ksaluja15.github.io/Learning-Rate-Multipliers-in-Keras/ 2) Adam - https://erikbrorson.github.io/2018/04/30/Adam-with-learning-rate-multipliers/

Both of these integrate nicely with the current Keras version. Thanks to the authors of these two implementations.

For the purpose of this excercise, we will use the data provided by the Kaggle Competition "Histopathologic Cancer Detection". The data for this can be downloaded with a Kaggle account at the following location: https://www.kaggle.com/c/histopathologic-cancer-detection/data

In this example, we are able to achieve 91% accuracy with no fine tuning of any hyperparameters.
