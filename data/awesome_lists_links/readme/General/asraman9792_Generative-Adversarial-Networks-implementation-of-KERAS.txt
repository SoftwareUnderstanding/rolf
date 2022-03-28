# Generative-Adversarial-Networks-implementation-of-KERAS
Implementation of Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. 

Implementation of Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.

Implementation:-

$ cd cyclegan/
$ bash download_dataset.sh apple2orange
$ python3 cyclegan.py

<p align="center">
    <img src="http://eriklindernoren.se/images/cyclegan.png" width="640"\>
</p>


The implementation of the Cycle GAN is for implementing the transformation of Apple to Orange and Orange to Apple.

Image-to-image translation is a class of vision and
graphics problems where the goal is to learn the mapping
between an input image and an output image using a training set of aligned image pairs. However, for many tasks,
paired training data will not be available. We present an
approach for learning to translate an image from a source
domain X to a target domain Y in the absence of paired
examples. Our goal is to learn a mapping G : X → Y
such that the distribution of images from G(X) is indistinguishable from the distribution Y using an adversarial loss.
Because this mapping is highly under-constrained, we couple it with an inverse mapping F : Y → X and introduce a
cycle consistency loss to enforce F(G(X)) ≈ X (and vice
versa). 

Paper: https://arxiv.org/abs/1703.10593


<p align="center">
    <img src="http://eriklindernoren.se/images/cyclegan_gif.gif" width="640"\>
</p>
