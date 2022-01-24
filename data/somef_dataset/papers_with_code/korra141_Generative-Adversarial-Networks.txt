Generative-Adversarial-Networks 

The idea originated from the paper published by Ian Goodfellow (https://arxiv.org/abs/1406.2661). From then on many varaitions have come up demonstating different versions of image clarity and model stabilty. 

In this repository three types of GAN have been implemented using python library tensorflow and sonnet namely Vanilla GAN (https://arxiv.org/abs/1406.2661), Wasserstein GAN (https://arxiv.org/abs/1701.07875), SoftMAX GAN (https://arxiv.org/abs/1704.06191). At present continuos MNIST images are used as a dataset for this repository, thus the architecture and hyper-parameters are set accordingly to give the best result.

evaluation_score.py is used by the GAN models to evaluate the images produced in terms of Inception Score and Frechet Inception Distance.
