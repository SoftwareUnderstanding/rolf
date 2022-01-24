# DCGAN

DCGAN is a Deep Convolutional Generative Adversarial Network.
The DCGAN is comprised of two neural networks pitted against each other.
The _Generator_ neural net learns to create images that look real while the _Discriminator_
learns to identify images that are fake. 

Over time the images start to resemble the training input more and more. 
The images begin as random noise, and increasingly resemble hand written digits over time. Below gif shows 100 epochs of training: 


![gan_gif](mnist/dcgan.gif)

#

[1] Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014.
(Full paper: http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)

[2] Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).
(Full paper: https://arxiv.org/pdf/1511.06434.pdf)

[3] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998.

