# How to run the code

* Install dependecy with `pip install -r requirements.txt`
* First step is to prepare our data using `Download and Prepare Data.ipynb`.
* `Data analysis.ipynb` we tried diverse Machine learning method with some feature extraction

## Architecture

The architecture used in this project is based on ResNet (https://arxiv.org/abs/1512.03385) with squeeze and excitation block (https://arxiv.org/abs/1709.01507). It is a light weight 1 dimensional ResNet with 9 layers.


## Data processing

The data processing used was choosen by hyperparameter search. The code to run the hyperparameter search can be found in `Hyperopt.ipynb`. 

## Generate submision


### Within subject

* Run `Submission Withing Subject.ipynb`

### Cross subject

* Run `Submission Cross Subject.ipynb`
