# Reproducibility Project - "A Simple Baseline for Bayesian Uncertainty in Deep Learning"
This repository contains the code used in our attempt at the [NeurIPS 2019 Reproducibility Challenge](https://reproducibility-challenge.github.io/neurips2019/), in reproducing the methods proposed in the paper ["A Simple Baseline for Bayesian Uncertainty in Deep Learning"](https://arxiv.org/pdf/1902.02476.pdf) by W. Maddox et al. This project also served as the final project of the course [DD2412](https://www.kth.se/student/kurser/kurs/DD2412?l=en) at KTH Royal Institute of Technology.

The code is implemented in TensorFlow 1.14 for Python3.


## File structure
```
+-- preprocess_data.py (preprocessing of CIFAR-10/100 and STL-10)
+-- train.py (loads a network architecture, runs SGD to train it and saves the learned weights)
+-- train_swag.py (loads a network architecture, runs SWA/SWAG-Diag/SWAG to train it and saves the learned weights)
+-- test.py (loads learned weights, runs the a standard test procedure and reports resulting metrics and plot data)
+-- test_swag,py (loads learned SWAG parameters, runs the SWAG test procedure and reports resulting metrics and plot data)
+-- utils.py (utility functions)
+-- networks/ (directory with all architectures)
    |   +-- vgg16.py (the vgg16 implementation)
+-- plotting/
    |   +-- reliability_diagram.py (takes plot data output by test.py or test_swag.py and produces reliability diagrams)
+-- weights/ (suggested directory for model weights, *not* tracked by git)
+-- data/ (suggested directory for datasets, *not* tracked by git)
```

## Getting started
To install all necessary dependencies for the implementation, run the command
```
pip install -r requirements.txt
```
To preprocess a dataset to the required format, use ´preprocess_data.py´. Example:   
```
python preprocess_data.py --data_path data/cifar-10-raw/ --train_frac 0.9 --valid_frac 0.1 --save_path data/cifar-10/ --data_set cifar10
```
To train a VGG-16 model using regular SGD, use `train.py`. Example:   
```
python train.py --data_path data/cifar-10/ --save_weight_path weights/ --save_checkpoint_path checkpoints/ --save_plots_path plots/
```
To train a VGG-16 model using SWA/SWAG-dIag/SWAG, use ´train_swag.py´. Example:   
```
python train_swag.py --data_path data/cifar-10/ --save_param_path weights/ --save_checkpoint_path checkpoints/ --save_plots_path plots/
```
To test a model trained with SGD, use `test.py`. Example:   
```
python test.py --data_path data/cifar-10/ --load_weight_file weights/sgd_weights.npz
```
To test a model trained with SWA/SWAG-Diag/SWAG, use `test_swag.py`. Example:   
```
python test_swag.py --data_path data/cifar-10/ --load_patam_file weights/swag_params.npz --mode swag
```

## Resources used in creating our implementation
* The reproduced paper - ["A Simple Baseline for Bayesian Uncertainty in Deep Learning"](https://arxiv.org/pdf/1902.02476.pdf)   
* A vital paper referenced by the authors - ["Averaging Weights Leads to Wider Optima and Better Generalization"](https://arxiv.org/pdf/1803.05407.pdf)
* TensorFlow implementation of VGG-16: https://www.cs.toronto.edu/~frossard/post/vgg16/.   
Pre-trained weights: https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz
* CIFAR-10/100: https://www.cs.toronto.edu/~kriz/cifar.html   
* STL-10: http://ai.stanford.edu/~acoates/stl10/
