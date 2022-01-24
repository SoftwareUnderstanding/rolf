# Test for Quantum: UNet semantic segmentation with PyTorch

This model was developed on a standard UNet architecture using a Pythorch and scorder loss function to automate core detection (semantic segmentation) on the '2018 Data Science Bowl' data

The data is available on the [Kaggle website](https://www.kaggle.com/c/data-science-bowl-2018/data).

## Usage
**Note : Use Python 3.7 or newer**

### Explore of initial data and preparation of working datasets

Download the first stage training and test datasets from the website [Kaggle website](https://www.kaggle.com/c/data-science-bowl-2018/data)
Run Jupiter nootbook 'quantum_test_data_explore.ipynb' to prepare working data for the developed model

The training set of images and the corresponding masks would be placed in subdirectories 'data/img_train/images' and 'data/img_train/masks', respectively
Test images would be placed in a subdirectory 'data/img_test/images'


### Train and test the model

Êun the script 'unet_segmentation_1.py'
Model hyperparameters can be changed by specifying them on the command line or by changing the default values ??in the script itself
The intermediate results of the model training will be available in the 'images_1class' folder, and the model chtckpoint parameters in the 'saved_models_1class' folder
Model test results will be available in the folder 'images_test_1class'
To continue training the model from a chtckpoint, you must specify the corresponding value of the hyperparameter 'epoch'
To obtain masks for the semantic segmentation of new images, you must:
- place new images in a folder 'img_test/images'
- select the tested version of the pretrained model (from the 'saved_models_1class' folder) by specifying the corresponding value of the hyperparameter 'epoch'
- set the value of the hyperparameter 'n_epochs' to 1
- the segmentation results will be available in the folder 'images_test_1class'


## Notes on memory

The model was trained on 670 images with a size of 128x128
Model parameters use approximately 50 MB of memory

## Support

Original paper by Olaf Ronneberger, Philipp Fischer, Thomas Brox: [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)

![network architecture](https://i.imgur.com/jeDVpqF.png)
