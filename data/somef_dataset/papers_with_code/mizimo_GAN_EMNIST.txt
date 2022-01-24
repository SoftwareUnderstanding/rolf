# Generative Adversarial Networks for generating Extended-MNIST samples
### Training
One fully connected layer and two convolution layers for both the generator and the discriminator were used. The Generative Adversarial Network ([arXiv:1406.2661](https://arxiv.org/abs/1406.2661)) was trained for *approximately* 2k epochs, over the *extended-MNIST dataset*, which contains handwritten digits as well as letters of the English Alphabet. 

The performance of the network seemed to be very sensitive to the hyper-parameters and initializations, especially the learning rate. Too large a learning rate caused the GAN to diverge from the minimum, whereas too small almost always led to Mode Collapse.

The use of xavier initializer and relu layer (non leaky) gave best results.

### Generated Samples
<img src="https://github.com/mizimo/GAN_EMNIST/blob/master/train.gif" width="50%">

*Fig. Samples generated during training*

### Usage
The network was trained on 23x19 sized EMNIST images. For usage on a slightly different dataset, the dimension of the input in main.py and the dimension of the hidden layers in lib/generator.py and lib/discriminator.py need to be changed. 
