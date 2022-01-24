# STAT923 Final Project
The code within was created during a cumulative project for a Multivariate Analysis project. The idea was to attempt to learn low-dimensional latent representations of images of faces in order to conduct facial verification tasks. In general different autoencoder structures were used, with a focus on variational autoencoders that can be shifted to generative models. The repository contains code for a standard Variational Autoencoder, the beta-Variational Autoencoder (which helps learn disentangled latent representations, by modifying the loss from a standard VAE), a VAE which is trained in batches and attempts to hold components constant for the purpose of learning disentangled representations, and a end-to-end CNN which attempts to do the classification without learning an explicit low-dimensional representation.

## Models
Any of the models should be able to have their relevant file taken in the directory, and be run separate from the rest of the repository. While the loading of data and the saving of weights has been specifically written for the project's application, modest modification should allow the same models to be used in other applications. In particular, changing the encoder and decoder structures, should allow for arbitrary model structures to be leveraged.

* `VAE`: This provides a standard variational autoencoder implementation (Kingma and Welling, 2013).
* `Beta-VAE/`: The Beta-VAE model (Higgins, et al. 2017) modifies the loss in a standard variational autoencoder through the addition of a single hyperparameter, which encourages the learning of disentangled latent embeddings. Ultimately, this is the most performant model for the task at hand that was compared for this project.
* `E2E-CNN/`: This serves as a base-line comparison for the task of facial verfication. It applies a very simplistic CNN pipeline to the image data directly, attempting to take a purely ML approach to the task; this is intended to serve as little more than a benchmark to see how well the explicit learning of a latent space compares.
* `ModVAE/`: This modifies the standard VAE in terms of both the loss-function, and the actual model structure.  The model takes ideas from work done by (Kulkarni et al., 2015), who attempted to enforce disentangled and interpretable factors by training the network with batched data, wherein each batch only varied along a single latent factor. The Deep Convolutional Inverse Graphics Network (DCIGN) that they proposed would, for instance, take in a set of images which were identical except for the angle of the light source; on this batch, they would effectively clamp all latent variables that they did not want to correspond to the location of the light source (i.e. clamp z2, · · · , zN ) and allow just the one to vary (i.e. z1). In so doing they effectively forced sets of latent variables to conform to the traits they desired, allowing the remaining variation to be learned in the remaining latents. In my modified VAE (ModVAE), I attempted a similar process, training the network on batches of images which varied along only a single axis. It uses standard forward propagation, then averages the output from all the encoder networks, and replaces all the encoded latent factors, which do not correspond to the latent factor which is allowed to vary, with the mean value. The loss function was altered slightly with an additional term added. This term measured the MSE between the predicted latent variables along the clamped axes and the mean values computed across the batch. The basic intuition here was to force all of the images to be encoded identically for each of the latent factors that they were identical for.
* `R Analysis`: This provides the R code that was used to take the multidimensional embedded vectors, and attempt to perform facial verification/multivariate analysis on them. This is done in R to conform to course requirements.

## Results
The results in terms of the capacity to conduct facial verification underperform other published methods, on the whole. However, the implemented models do have a strong ability to re-construct faces from the relevant latent spaces, and to generate new faces which are not based on existing references images.

![Reconstructed Images](Reconstructed.png)

In this figure we the top row are images which appear in the dataset directly. The following 4 rows represent the reconstruction attempts by β-VAE with β = 2, 4, 5, 10 respectively. We can see that each network is able to reproduce the images to varying extents.

![Generated Images](generated_images.png)

Here we have a selection of generated images from the various models. Each image was generated merely by selecting a 32 dimensional standard normal vector in the latent space, then feeding it to the trained model. None of these images are direct reconstructions of images from the dataset.

## References
Diederik P Kingma and Max Welling. “Auto-Encoding Variational Bayes”. In: Ml (2013), pp. 1–14. issn: 1312.6114v10. doi: 10.1051/0004- 6361/201527329. arXiv: 1312.6114. url: http://arxiv.org/abs/1312.6114.

Irina Higgins et al. “beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework”. In: Iclr July (2017), pp. 1–13. url: https://openreview.net/forum?id=Sy2fzU9gl.

Tejas D. Kulkarni et al. “Deep Convolutional Inverse Graphics Network”. In: (2015). issn: 10897550. doi: 0.1063/1.4914407. arXiv: 1503.03167. url: http://arxiv.org/abs/1503.03167.