# Image Noise Removal

![](https://github.com/fescobar96/Image-Noise-Removal/blob/master/Images/Picture1.png?raw=true)

## Introduction

Similar to generative adversarial networks, autoencoders work by learning the latent space representation of the data that is fed in. The latent space contains all the necessary information to recreate the features of the original data. 

In this project, I used an autoencoder to remove salt and pepper noise from corrupted images. It is important to highlight that this type of neural network could be easily applied to another type of problems like increasing resolution of images, image colorization, film restoration, and background removal.



## Dataset

For this project, I used the 'Natural Images' dataset, which was created as a benchmark dataset for the paper: *'Effects of Degradations on Deep Neural Network Architectures'* by *Roy et al.* The dataset is composed of 6,899 images of the eight following classes: airplanes, cars, cats, dogs, flowers, fruits, motorcycles, and people. A more diverse dataset could have yield better results and a more reliable and widely applicable model.



## Methodology

The first step of this project was to generate 'corrupted' images. To achieve this, the I added a random amount of salt and pepper noise to every image in the dataset and saved it in a separate folder from the original images. For more details, please open the **Noisy_Images_Generator.ipynb** notebook.

The denoising process started by importing in a sorted fashion both the 'corrupted' images and the clean images. It is important to import the images in order because the cleaned dataset will be used by the autoencoder as a reference of what the noisy images should look like. After importing and normalizing the images, the two datasets were split into a training (80% of data) and validation set (20% of data.) 



The autoencoder is composed of 9 layers: 

- **Layers 1-4:** The first 4 layers are Conv2D layers with batch normalization, Leaky ReLU activation, and a decreasing number of filters.
- **Layers 5-8:** The next 4 layers are Conv2DTranspose layers which could be considered the inverse of a Conv2D layer. These layers also have batch normalization, Leaky ReLU activation, but unlike the first four layers, the numbers of filters is increasing on each layer.
- **Layer 9:** Finally, the output layer is a Conv2DTranspose layer with sigmoid activation.



The model that I present differs from a traditional deep convolutional autoencoder mainly on the use of batch normalization and Leaky ReLU. The rationale behind my improvements is the following:



- **Batch Normalization:** By normalizing the inputs of each layer for each batch, I can limit how much the distribution of the inputs received by the current layer is affected by the previous layer. This decreases any possible interdependence between different parameters, and it helps increasing the training speed and the reliability of the model.
- **Leaky ReLU:** Unlike the ReLU activation function, Leaky ReLU can prevent gradients from dying by allowing a small positive gradient instead of zero. 



## Results

The results were satisfactory, but they are still far from being perfect. The most notorious issue is the blurriness and discoloration of the output images. The model preforms in a suboptimal way when the amount of noise in an image is relatively low, however, it delivers impressive results when the image is highly noisy and corrupted. The images below were randomly drawn from the validation set.

![](https://github.com/fescobar96/Image-Noise-Removal/blob/master/Images/Picture1.png?raw=true)

![](https://github.com/fescobar96/Image-Noise-Removal/blob/master/Images/Picture2.png?raw=true)

![](https://github.com/fescobar96/Image-Noise-Removal/blob/master/Images/Picture3.png?raw=true)

![](https://github.com/fescobar96/Image-Noise-Removal/blob/master/Images/Picture4.png?raw=true)

## References

1. Ioffe, Sergey, and Christian Szegedy. “Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.” *ArXiv:1502.03167 [Cs]*, Mar. 2015. *arXiv.org*, http://arxiv.org/abs/1502.03167.
2. Roy, Prasun, et al. “Effects of Degradations on Deep Neural Network Architectures.” *ArXiv:1807.10108 [Cs, Eess]*, June 2019. *arXiv.org*, http://arxiv.org/abs/1807.10108.