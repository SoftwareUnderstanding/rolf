# CycleGANs

Generative adversarial networks (GANs) are algorithmic architectures that use two neural networks, pitting one against the other (thus the “adversarial”) in order to generate new, synthetic instances of data that can pass for real data. 

CycleGAN is a technique that involves the automatic training of image-to-image translation models without paired examples. The models are trained in an unsupervised manner using a collection of images from the source and target domain that does not need to be related in any way.

This project aims to reproducing the results from the paper "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" by Jun-Yan Zhu et al. The model converts an image from a particular domain to another domain. <br />
Link to the paper: https://arxiv.org/pdf/1703.10593.pdf

For example, Zebra - Horse, Apple - Orange, Satellite view - Map view, MNIST - color inverted MNIST, Summer Landscape - Winter Landscape. Some more applications: https://lnkd.in/eWBq7bX
