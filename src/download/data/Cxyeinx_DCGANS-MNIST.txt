# Deep Convolutional Adversarial Network

This is a generative model using Convolutional Neural Networks for generating various things using noise.

The main paper on this can be seen here: https://arxiv.org/pdf/1511.06434.pdf

The things to note in this paper: 
1. Spartial Pooling functions (eg. Max Pooling) was replaced with Strides.
2. Dense layers aren't used.
3. The last layer of Discriminator uses sigmoid activation function.
4. Batch Normalization is used but not to all layers (generators output and discriminators input), as it resulted in model instability.
5. ReLu is used in generator with the output as Tanh.
6. LeakyReLu is used in discriminator with the output as Sigmoid.
7. Batch size of 128 was used.
8. Adam as the optimizer was used with 0.0002 learning rate and 0.5 as beta 1.
9. Noise dimension was 100.

Lets hope it works, if it didn't.
Don't loose hope, you have your whole life to adjust hyper params. 


![](results/movie.gif)
