# MangaGen
A DC-GAN trained on anime image on Kaggle.

Dataset: https://www.kaggle.com/splcher/animefacedataset

The dataset consists of over 60,000 images which are used to train this DC-GAN.

# Training Result

Do keep in mind that this is trained on an image set which is 64 x 64, so the actual images have to be viewed at a much smaller size. This means that the GAN is actually performing better given the constraints. For now, a larger scaled version is shown which causes pixelation to be a lot more apparent.

![Image of Showcase of Results](https://github.com/tejjogani/MangaGeneration-GAN/raw/master/results/showcase.gif)



# Further Details
The network architecture used is inspired by a paper on DC-GANs named UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS Reference: https://arxiv.org/pdf/1511.06434.pdf

The following image represents the network architecture for the Generator G(Z):

![Image of Network](https://github.com/tejjogani/MangaGen/raw/master/resources/network.png)

Naturally, the Discriminator Architecture would be reversed with Transposed Convolution layers sandwhiching layers that perform BatchNormalization.

# Tuning Hyperparameters

Currently, no method to add a parser has been used, and the hyperparamters are set to their optimal values via experimentation for this particular dataset. However, in consequent commits, a parser should be made for ease of use in the terminal, along with a pretrained model for this dataset. The current hyperparamters are:

```
learning_rate = 0.0002
batch_size = 128
image_dimensions = 64
num_epochs = 20
```
The current loss function being used for the generator is BCELoss (Binary Cross-Entropy Loss) as one would with a DC-GAN.
