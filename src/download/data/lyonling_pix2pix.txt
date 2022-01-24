# pix2pix

[TOC]

###  Datasets:

<https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/>

### Reference:

* <http://sse.tongji.edu.cn/linzhang/CV/ReadingMaterials/Image-To-Image_Translation.pdf>
* <https://arxiv.org/pdf/1703.10593.pdf>

## Proposal

### 1. Background

In the image processing domain, the targets of many problems are always to convert one input image to its corresponding output image, like the transformation among greyscale image, gradient image and coloured image. Generally, specified algorithms will be applied for different question. For example, when using CNN to solve the problem of image translation, it's needed to define the loss function for the optimization of each problem. A common solution is to narrow down the Euclidian distance between input and output data by training CNN, while the fuzzy output would be get for the most situations. 

The essence of these issues is just constructing the mapping between pixels to pixels. Therefore, based on the theory of GAN(Generative Adversarial Network), pix2pix, as a common method was raised to solve these problems.

### 2. Related Work

**Generative Adversarial Network:** Basically, we mentioned Generative Adversarial Network[1], GANs learn a loss that tries to classify if the output image is real or fake, while simultaneously training a generative model to minimize this loss. From which we can generate target images by feeding in some dummy input data. However, traditional GAN is not powerful enough to solve all of our image processing tasks, below we conclude two drawbacks of GAN. Lack of user control abilities. Which means user could not specify the details of outputs during training and testing, except for adjusting the training dataset.
For the most situation, basic GAN could not generate images with high quality. Even if the output image looks pretty good, it would be vague when zooming in because of low resolution. 

**Conditional GANs**: As we known that if user want to make their GANs more controllable when doing image generation, some constraints should be applied on the generator network. Prior and concurrent works have GANs on discrete labels, text, and images[2]. In which specified high-degree vector would be train to form the conditions or constraints for input data. This kind of image-conditional models could handle  image prediction from a normal map, future frame prediction, product photo generation, and image generation from sparse annotations.

**Pix2Pix:** Several other papers have also used GANs for image-to-image mappings, but only applied the GAN unconditionally, relying on other terms (such as L2 regression) to force the output to be conditioned on the input. One of those work we refer to followed the shape of U-net to build generator and applied Markovian discriminator to implement the image to image transformation with paired data input[3]. Our idea is based on the form of pix2pix GANs, some adjustments could be workable in the oringin network structure to improve the model efficiency.

### 3. Our Work

### 4. Reference

- [1] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial nets. In NIPS, 2014.
- [2] M. Mirza and S. Osindero. Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784, 2014. 
- [3] Isola P , Zhu J Y , Zhou T , et al. Image-to-Image Translation with Conditional Adversarial Networks[J]. 2016.



