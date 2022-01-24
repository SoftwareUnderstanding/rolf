# Generating fake faces with using Deep Convolutional Generative Adverserial Networks

In this project we've used Deep Convolution Generative Adverserial Network, or DCGAN for short, and a custom dataset, in order to generate new Faces.

### Prerequisites

Before trying to run any of these programs you will need to install a couple of things in the following order:
1. Anaconda ( latest version ) 
  - Link : https://www.anaconda.com/download/
2. Python Lybraries such as : PyTorch, NumPy, Matplotlib, etc.
  - After installing Anaconda you will be able to use their open source package and system managment system, also
  known as "conda". Here is a link on how to download Pytorch using "conda" package managment system : https://anaconda.org/conda-forge/pyroch

## Running the project

After you've completed steps specified above, you will be able to run the program on your machine. I suggest you use Spider programming enviroment . To open Spyder and run the program you have to:
 - Open Anaconda Navigator
 - Launch your Spyder3 Enviroment.
 - Position yourself on directory where you've downloaded the project using file explorer tab located in the top right corner of the enviroment. 
 - Run main.py

## Results 

Results are not as satisfying as we were expecting. We are still experimenting. Currently we are trying to figure out why does SELU activation function give poor results even though we've followed the specified architecture implementation from the "High-Resolution Deep Convolutional Generative Adverserial Networks" paper. 

## Authors

* **Stefan Stamenkovic**
* **Damjan Nicic** 

## Acknowledgment
- GANs, original paper by Ian J. Goodfellow: https://arxiv.org/abs/1406.2661
- DCGAN , paper by Alec Radford & Luke Metz: https://arxiv.org/abs/1511.06434
- HDCGAN, paper by Joachim D. Curto, Irene C. Zarza, Fernando De La Torre, Irwin King, and Michael R. Lyu: https://arxiv.org/pdf/1711.06491
- GANs: Overview, paper by Antonia Creswell, Tom White, Vincent Dumoulin, Kai Arulkumaran, Biswa Sengupta and Anil A Bharath : https://arxiv.org/pdf/1710.07035
