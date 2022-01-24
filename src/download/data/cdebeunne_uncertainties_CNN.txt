# Uncertainties Neural Networks

A notebook that implements a few toy neural networks in order to test uncertainties estimation.

- A 1D regression with the CO2 dataset
- An image classifier for MNIST's data
- A custom pose estimator for 2D squares, inpired by Yann Labbe's 6D object tracker cosypose https://github.com/ylabbe/cosypose 

Two methods are explored in the associated notebook:

- Monte Carlo Dropout : method theorized and explained in https://arxiv.org/pdf/1506.02142.pdf (Gal 2016)
- Assumed Dense Filtering : inspired by Gast et al. article https://arxiv.org/abs/1805.11327

The code of ADF's function (contrib folder) is provided by Mattia Segù and available @ https://github.com/mattiasegu/uncertainty_estimation_deep_learning
