# Pix2Pix GAN model
image-to-image-translation

In this project, Pix2Pix GAN model was utilized for image-to-image translation of SAR because Pix2pix achieves the desired results for many image-to-image translation tasks. for implementation of pix2pix model, TensorFlow and Keras deep learning framework were used. 
The architecture consists of two models, namely a generator model which is an encoder-decoder model using a U-Net architecture, that means that skip-connections are added between the encoding layers and the corresponding decoding layers, to produce new acceptable combining different artificial images, and a discriminator model to classify images in two groups as they are real images (from the dataset) or fake ones (generated) that  is a deep convolutional neural network performing image classification, conditional-image classification in particular. This is called a PatchGAN model. 
In the GAN model, the generator model and discriminator model are trained in adversarial process at the same time, in this way, the discriminator model is updated directly, while the generator model is updated through the discriminator model. 

## Generative Adversarial Network (GAN)
Plenty of deep learning techniques have been employed to SAR image processing being able to cope with the tasks of image-to-image translation which are formulated as per-pixel classification or regression. Advances in deep learning methods have led to a breakthrough in deep learning that Generative Adversarial Networks (GANs) given that as one of them. In fact, GANs are capable to find the internal data distribution using a large number of data. In spite the fact that GANs generate data in an unsupervised, they are weak to handle the data generation process, therefore, to tackle this problem, other type of GANs called the Conditional Generative Adversarial Networks was proposed. In this way, through conditioning the model on extra information, cGANs can direct data generation process.

A type of Conditional GAN or cGAN model called Pix2Pix, has been utilized for image-to-image translation task. Pix2Pix model as the well-known and powerful supervised model, widely used for image-to-image translation. Recently, this method is being dealt to the visual observation problem of SAR images. 

## Synthetic-aperture Radar (SAR) 

Remote sensing satellites collect data with different spatial and spectral characteristics around the world indicating characteristics of features on Earth. Sometimes the information of one sensor cannot meet our needs. Despite the fact that multi-spectral data provide significant spectral information form features, they are significantly affected by environmental factors such as smoke, fog, clouds, and sunlight. In this case, Synthetic-aperture Radar (SAR) sensors in comparison with optic sensors are able to get data in any meteorological and atmospheric conditions. Thanks to deep learning image translation models, we are able to convert SAR images to optical RGB image that are more directly comprehensible.

## Dataset
Download pix2pix datasets from: https://www.kaggle.com/vikramtiwari/pix2pix-dataset

## Reference
https://arxiv.org/abs/1611.07004

https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix?utm_source=catalyzex.com

https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/
