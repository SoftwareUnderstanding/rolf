# FocusingNeuron

Repo for focusing neuron (adaptive locally connected neuron)

Paper: https://arxiv.org/abs/1809.09533
Notes: 
1) Paper is an older version with slightly different focus function normalization 
2) Current code can provide even better results. 



## Code

Depends on other libraries: numpy, scikit, theano, lasagne

#### UPDATE Nov 2019 : Keras version is transferred to another REPO. 

https://github.com/btekgit/FocusingNeuron-Keras



#### Some experiment jupyter notebooks are provided in experiment-notebooks folder. 
To run in Google colab you must upload Kfocusing.py and keras_utils.py for Keras based.



### EXAMPLES

- Quick example runs on synthetically generated classication datasets:
*python Test-Synthetic-Inputs.py

- MNIST example

*python mnist.py focused_mlp:2,800,0.25,0.25 10 1 mnist mnist10 0.0

Test set accuracy is ~99.10-99.20

*python mnist.py mlp:2,800,0.25,0.25 10 1 mnist mnist10 0.0

Test set accuracy is ~98.9-99.05


### DATA 

Requires mnist.npz or downloads it from http://yann.lecun.com/exdb/mnist/
Other datasets such as cifar_10 and fashion can be downloaded with keras.datasets
Note: mnist_cluttered data is difficult to find in internet again. Email me if you cant find it. I will upload it 



### EXPERIMENTS
Repeated trial experiments are implemented .sh files. Contains my local directory references.


### UPDATE AUG 2019:
I have added keras implementations and some new ipynb for experiments:
- Kfocusing: the focusing neuron layer class file, include a unit test (Requires included keras_utils.py)
- KfocusingTransder: test focusing neuron in transfer learning with keras.applications  and pretrained models (VGG1-16)
- Boston experiment notebook
- Reuters experiment notebook (however, theano and python version worked better)

- Random_Syntethic_Tests-master-forPaper-ready.ipynb  repeats the synthetic experiments

- Focusing_Network_Test_Single_Run-Mnist-For-Paper-ready.ipynb repeats a single run MNIST experiment

- USE dense-nn-weights-mnist-eng-ready.ipynb to experiment on Dense network Weights with noise PADDED MNIST

- NOTE Keras versions can be run in GOOGLE colab


