# StyleGAN2 with Pytorch

Simple implementation of **Analyzing and Improving the Image Quality of StyleGAN** ([http://arxiv.org/abs/1912.04958](http://arxiv.org/abs/1912.04958)).

The project is dedicated to the rapid prototyping of StyleGAN2-like networks using the building blocks defined here. It includes
- modulated conv layers
- minibatch std layer
- mapping network
- noise with learnable magnitude
- standard blocks for generator and discriminator
- equalized lr
- scaled activations
- losses
- projection method

If you need a complete network as in the paper you can easily implement it knowing all the parameters and dimensions. See the toy example [here](example.ipynb).

### Example on Mnist:

![mnist_example](samples/mnist.png)

I would appreciate help with testing.

### TODO

- [ ] Base classes for Generator and Discriminator 
- [ ] Arrange the training procedure.
