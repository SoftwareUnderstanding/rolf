# Cyclegan-Tensorflow

Code for reproducing experiments in (https://arxiv.org/pdf/1703.10593.pdf) - Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks

It is an improved version which include checkpoints so you can stop and start it from same point. I also included tensorboard fro getting the loss functions.
The image generated is of two types 
	1. is comparing with the oiriginal 
	2. is single fake image generated

### Prerequisites

* Python, NumPy, TensorFlow, SciPy, Matplotlib, pillow, Keras, Tensorboard
* Better if you have a GPU


### Installing

First step is having a dataset
Dataset that can directly work -
	* maps
	* facade
	* night2day
	* edge2shoes
	* edge2handbages

After downloading the dataset, You need to put dataset in dataset folder
and set the dataset name in main.py 

You can change epoch, batch size, print frequency(for image generation), image size and learning rate in mani.py as well

after that you go and run main.py

```
python3 main.py
```
