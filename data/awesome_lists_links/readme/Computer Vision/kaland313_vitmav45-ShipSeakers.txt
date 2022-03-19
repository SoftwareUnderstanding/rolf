# Ship detection on satellite images using deep learning
The scripts found in this repository were written as the project assignment for the course Deep Learning in Practice with Python and LUA (VITMAV45). As part of this project, we aim to provide a solution for the Airbus Ship Detection Challenge organized by Kaggle. More information about the competition can be found on its website: https://www.kaggle.com/c/airbus-ship-detection

Team name: ShipSeakers

Team members:
* Csatlós, Tamás Péter (cstompet@gmail.com)
* Kalapos, András (kalapos.andras@gmail.com)


## Motivations
Earth observation using satellite data is a rapidly growing field. We use satellites to monitor polar ice caps, to detect environmental disasters such as tsunamis, to predict the weather, to monitor the growth of crops and many more. 
*Shipping traffic is growing fast. More ships increase the chances of infractions at sea like environmentally devastating ship accidents, piracy, illegal fishing, drug trafficking, and illegal cargo movement. This has compelled many organizations, from environmental protection agencies to insurance companies and national government authorities, to have a closer watch over the open seas.* (Quoted from the description of the kaggle challenge)

## Goals, description of the competition
We are given satellite images (more accurately sections of satellite images), which might contain ships or other waterborne vehicles. The goal is to segment the images to the "ship"/"no-ship" classes (label each pixel using these classes). The images might contain multiple ships, they can be placed close to each other (yet should be detected as separate ships), they can be located in ports, can be moving or stationary, etc. The pictures might show inland areas,the sea without ships, can be cloudy or foggy, lighting conditions can vary. 
The training data is given as images and masks for the ships (in a run length encoded format). If an image contains multiple ships, each ship has a separate record, mask. 

## Prerequisites
The training data can be downloaded from the competition's website after agreeing to the terms. Please note that the data might not be available after the submission deadline. 
We used Python 3.6 with Keras and Tensorflow and some other necessary packages. 

## Directory structure and files
The folder train_img contains all the images downloaded as the train_v2.zip
```
├── ShipDetectionDataPrep.ipynb
└── data/ - sample data
    ├── train_ship_segmentations_v2.csv
    └── train_img/
└── Model, Training and Evaluation/Scripts/
    ├── Train.py -  Script used to train the model
    ├── ShipSegFunctions.py - a file containing important multi-use functions, sort of a project library
    ├── model.hdf5 - The model saved with complete structure and weights
    ├── model.png - Plot of the model structure
    ├── Evaluation.ipynb - The result visualization and evaluation notebook
    ├── Img
        └── Test images with ground truth and predicted segmentation maps (in separate files
    ├── Training history
        └── Various files showing the training history, log 
    ├── train_img_ids.npy - a file binary storing image file names of the training partition
    ├── valid_img_ids.npy - a file binary storing image file names of the validation partition
    └── test_img_ids.npy  - a file binary storing image file names of the test partition
```

## Data exploration and preparation script - Milestone I
The training data is analysed and visualised in  the [ShipDetectionDataPrep.ipynb](ShipDetectionDataPrep.ipynb) Jupyter Notebook.
The script can be executed if a few images are included in the train_img folder. Whithout this it can't show examples for different scenarios apperaring in the dataset. 

## Training
The ShipSegFunctions.py script mostly conains functions and the generator class which are are to run the train and test scripts.

The Train.py sript loads and preprocesses the data for training, it includes network definitions and the training. The dataset is split into training and test partitions, but the testing is done in a separate file so the IDs of the test dataset are stored in test_img_ids.npy. The model is saved to model.hdf5 after every epoch which improves on the network performance.

## The trained network
The trained network is a typical Unet architecture network, presented in the paper by Ronneberger (https://arxiv.org/abs/1505.04597), which is widely used for semantic segmentation. 
The structure of network is presented on the  [model.png](Model%2C%20Training%20and%20Evaluation/Scripts/model.png). 

## Results 
The large dataset containing more than a hundred thousand marine satellite images was analysed and
a deep fully convolutional network based solution to perform semantic segmentation of ships on this
images was presented. The network based on the U-Net architecture was successfully implemented,
trained and tested. The best achieved dice coefficient on our test set was 0.714.

The evaluation and visualisation of the network performance is presented in the [Evaluation.ipynb](Model%2C%20Training%20and%20Evaluation/Scripts/Evaluation.ipynb) Jupyter Notebook. 

One prediction the network made can be seen on the picture below. 
![Example predictions](https://github.com/kaland313/vitmav45-ShipSeakers/blob/master/ExamplePrediction.png)

More predictions can be found in the [Evaluation.ipynb](Model%2C%20Training%20and%20Evaluation/Scripts/Evaluation.ipynb) Jupyter Notebook and some full resolution images and corresponding segmentation maps (ground truth and predicted) are located in the [Img folder](Model%2C%20Training%20and%20Evaluation/Scripts/Img). 

## Report
A written report in English (including an abstract) is located in the [Report](Report) folder: [ShipSeakersReport.pdf](Report/ShipSeakersReport.pdf).

The Hungarian abstract is can be founc in [ship_seakers_abstract_hun.pdf](Report/ship_seakers_abstract_hun.pdf)
