# Dcgan-Tensorflow

Code for reproducing experiments in (https://arxiv.org/pdf/1511.06434.pdf) - UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL
GENERATIVE ADVERSARIAL NETWORKS.


It is an improved version which include checkpoints so you can stop and start it from same point. I also included tensorboard fro getting the loss functions.

### Prerequisites

* Python, NumPy, TensorFlow, SciPy, Matplotlib, pillow, Keras, Tensorboard
* Better if you have a GPU


### Installing

First step is having a dataset
Dataset that can directly work -
	* Mnist
	* Celeba
	* Cifar10
	* stl10
	* LSun(Bedroom)

After downloading the dataset, You need to put dataset in dataset folder
and set the dataset name in main.py 

You can change epoch, batch size, print frequency(for image generation), image size and learning rate in mani.py as well

after that you go and run main.py

```
python3 main.py
```
