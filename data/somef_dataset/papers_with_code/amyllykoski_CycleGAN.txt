# CycleGAN

This is a Jupyter notebook containing a deep learning project about Generative Adversarial Network, namely CycleGAN.
The objective is to generate images of certain style using syntethically generated data as an input.

1. Clone the repository and navigate to the downloaded folder.

	```
	git clone https://github.com/amyllykoski/CycleGAN
	cd CycleGAN
	```

2. Use your own dataset (trainA, trainB, testA, testB) as outlined in https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

3. Make sure you have already installed the necessary Python packages like so:

    	pip install -r requirements.txt

Open a terminal window and navigate to the project folder. Open the notebook and follow the instructions.

	jupyter notebook .
	
# PointNet

Adapting support for point cloud CNN from 

```
https://github.com/romaintha/pytorch_pointnet
https://www.qwertee.io/blog/deep-learning-with-point-clouds/
https://arxiv.org/pdf/1612.00593.pdf

```
The objective is to see if CycleGAN can be adapted to work with point cloud based (depth) images.
