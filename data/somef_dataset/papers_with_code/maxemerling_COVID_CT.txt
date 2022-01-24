# BayesianCT

## Probability Visualization
BayesianCT.ipynb is a jupyter notebook that lets you interactively visualize the effects of testing a patient for COVID-19 using RT-PCR and CT-Scan tests (or any others you define). Given the sensitivity, specificity range, and results of the tests you define, the visualization will provide the probability range that a patient is infected.  

The math is fairly simple, relying on Bayes's Theorem and the Law of Total Probability, but the notebook could nonetheless be helpful for the early decision stages of whether the benefits of an immediate CT scan in addition to an RT-PCR test (which can only be performed every 3 days) are worth the costs (price, radiation exposure, etc.).

## Modeling CT Scan Tests
In the hopes of attaining values for the specificity and sensitivity of a CT-Scan test (and validating the source liked below), I'm attempting to train a baseline model for detecting COVID-19 pneumonia.  

To get a baseline for this type of classificiation, I'm first training a model to simply distinguish between healthy chest X-rays and those with pneumonia (both viral and bacterial).  

For the sake of efficiency, I'm starting with a SqueezeNet implementation in Keras (keras_squeezenet.py). Given that the problem is binary classification, I'm guessing larger, higher-accuracy models like ResNet won't be necessary. However, if SqueezeNet demonstrates limited performance, I may try MobileNetV2, which is slightly larger than SqueezeNet but also achieved slightly better performance on ImageNet. I'll compare the results of randomizing weights and transfer learning with imagenet weights. The most up-to-date training pipeline is in xray_model.py.

## References
SqueezeNet Original Paper:  
https://arxiv.org/abs/1602.07360

Chest X-Ray Images (Pneumonia vs Healthy):  
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

Neural net classification of COVID vs bacterial pneumonia:  
https://pubs.rsna.org/doi/10.1148/radiol.2020200905
