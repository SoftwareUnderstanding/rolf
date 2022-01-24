# GAN-Reading-List

* *DCGAN* - [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) - Alec Radford, Luke Metz, Soumith Chintala. *arXiv* preprint arXiv:1511.06434. 2015.
  * important improvements of the architecure: conv with stride of 2 (instead of maxpool), use of batchnorm in generator and discriminator, elemination of the fully connected layers, use of LeakyReLU in discriminator and ReLU in generator
  * example images on LSUN after 1 and 5 epochs
  * evaluation of DCGAN as a feature extractor by using the discriminator's conv. layers together with an SVM 
  * visualizations of generator - walking in the latent space: small changes on the manifold to show changes to the image generations
  * visualizations of discriminator - guided backprop
  * evaluated on CIFAR-10, SVHN, LSUN, Faces (new dataset, 350K face boxes)

* [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498) - Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen. 
  * feature matching: new objective for the **generator** - feature extracted from intermediate layer of discriminator should match (ie output of generator with real data distibution)
  * minibatch discrimination: combat collapse of generated features -> allow discriminator to look at multiple data points, 
