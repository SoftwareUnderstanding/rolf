# FocusingNeuron-Keras
Keras TF implementation of Adaptive Locally Connected Neuron: FocusingNeuron

Repo for focusing neuron (adaptive locally connected neuron)

Paper: https://arxiv.org/abs/1809.09533 Notes:
    
## UPDATE AUG 2019

I have added keras implementations and some new ipynb for experiments:

    Kfocusing: the focusing neuron layer class file, includes tests for simple 2-hidden layer Focused network and CNN+FCS architectures.  (Requires included keras_utils.py)

    KfocusingTransder: test focusing neuron in transfer learning with keras.applications and pretrained models (VGG1-16)

    Boston experiment notebook

    Reuters experiment notebook ()

    Random_Syntethic_Tests-master-forPaper-ready.ipynb repeats the synthetic experiments

    Focusing_Network_Test_Single_Run-Mnist-For-Paper-ready.ipynb repeats a single run MNIST experiment

    USE dense-nn-weights-mnist-eng-ready.ipynb to experiment on Dense network Weights with noise PADDED MNIST

    NOTE Experiments notebooks can be run in GOOGLE colab

## UPDATE APRIL 2020
    I have added tensorflow 2. folder which includes tf2.keras codes. 
    This version is working but optimizers are not 100%. I can't set
    separate learning rates for MU and Sigma. 

    In addition, I have weight-share option. The neurons can share their
    weights now. sharedWeights parameter in FocusedLayer1D now controls
    number of distinct weight sets (if it is 0, nweights=num_inputs x num_units)
 
