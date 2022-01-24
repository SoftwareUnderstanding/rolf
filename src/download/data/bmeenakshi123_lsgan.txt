# LSGAN
This is my implementation of the Least Squares General Adversarial Networks. This is the link to the research paper. https://arxiv.org/abs/1611.04076. This implementation uses MNIST dataset

Regular GANs hypothesize the discriminator as a classifier with the sigmoid cross entropy loss function.  This loss
function, however, may lead to the vanishing gradient problem during the learning process.  To overcome such problem, the research paper referenced above proposes the Least Squares Generative Adversarial Networks (LSGANs) that adopt the least squares  loss  function  for  the  discriminatorLSGANs are able to generate higher quality images than regular GANs. Second, LSGANs performs more stable during the learning process

The model and training function is in notebook - lsgan.ipynb.