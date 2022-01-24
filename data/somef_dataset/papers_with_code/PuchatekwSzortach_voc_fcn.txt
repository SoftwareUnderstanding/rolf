# voc_fcn
A Fully Convolutional Network implementation trained on Pascal Visual Object Classes dataset.

This work provides a tensorflow (version 1.8) implementation of "Fully Convolutional Networks for Semantic Segmentation" paper. There are some differences between this implementation and original paper - model is not trained in stages, and dense layers from VGG model are not used.

## Entry points
There are three main entry points in this project:  
- scripts/train.py - for training the model  
- scripts/analysis.py - for checking model's performance  
- scripts/visualize.py - contains a number of functions for visualizing dataset and predictions  

All scripts require path to configuration file as argument. **configuration.yaml** provides a sample configuration, as described in **configuration** section.


## Dataset
Following original paper, this project uses two datasets: [VOC2012 PASCAL] (http://host.robots.ox.ac.uk/pascal/VOC/voc2012) dataset and Hariharan's [extended VOC2011 PASCAL dataset](http://home.bharathh.info/pubs/codes/SBD/download.html). Note you will need to download train_noval.txt for Hariharan's set separately from dataset's page.


## Configuration
Paths to datasets, training hyperparameters, logging path, and many other settings are controlled through a configuration file. **configuration.yaml** provides a sample implementation.

## Results
We present three sets of images to show some of the best results, average results and failure cases.

#### Good results
![alt text](./images/good.jpg)

#### Average results
![alt text](./images/average.jpg)

#### Failures
![alt text](./images/bad.jpg)

Mean intersection over union of the best trained model for the validation dataset was **0.447**, with performance on each category provided in table below.

Category | Intersection over union
--- | --- 
aeroplane | 0.58477
background | 0.86000
bicycle | 0.19416
bird | 0.53809
boat | 0.27445
bottle | 0.36764
bus | 0.62230
car | 0.60776
cat | 0.59874
chair | 0.15179
cow | 0.31397
diningtable | 0.27226
dog | 0.52583
horse | 0.38766
motorbike | 0.48593
person | 0.66646
pottedplant | 0.28333
sheep | 0.45254
sofa | 0.25164
train | 0.53669
tvmonitor | 0.41259
