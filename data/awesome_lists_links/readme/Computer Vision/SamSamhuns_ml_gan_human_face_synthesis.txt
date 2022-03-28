# Generative Adversarial Networks: Face Synthesis

Implementation of a GAN for face synthesis using a celebrity image dataset.

## General Setup

### Set up Virtual Environment

Note: `Anaconda` can also be used for the venv

```shell
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

To start a new Jupyter Notebook kernel based on the current virtual environment:

```shell
$ python -m ipykernel install --user --name ENV_NAME --display-name "ENV_DISPLAY_NAME"
```

## Data

We use the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset to train our GAN. CelebA is a large-scale face attributes dataset with more than **200K** celebrity images.

![](readme_img/img/dataset_images.png)

### Download the CelebA Dataset

**Warning: Data size exceeds 70 GB**

`$ python download.py`

## Preprocessing Data

We the following transformations on image data so all images are cropped to size `64x64x3` and normalized:

    torchvision.transforms.CenterCrop(160)
    torchvision.transforms.Resize(64)
    torchvision.transforms.ToTensor()
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

## Base GAN Network

### Base Generator

Base model structure

![](readme_img/img/generator.png)

### Base Discriminator

Base model structure

![](readme_img/img/discriminator.png)

### Loss function

Binary Cross Entropy Loss for the minimax GAN loss

![](readme_img/img/minimax_loss.jpeg)

We use a `batch size` of `64`, latent vector z size of `100`

### Hyper Parameters used

```python
input image dimension = 64 x 64 x 3
latent vector z size  = 100
ADAM optimizer beta1  = 0.5
learning rate         = 0.0002
batch size            = 64
epochs                = 100
```

## Improved Model

In the improved model, based on other research work done on improving GANs, we change a few things in the Generator and Discriminator networks structures.

For the **Discriminator**, we add `nn.Dropout()` layers to add regularization and prevent overfitting.

For the **Generator**, we change the activation functions to `LeakyReLU()` instead of `ReLU()` to avoid sparse gradients.

In the Training itself, we add **One side label smoothing** to further regularize the Discriminator. i.e. Change the real labels from all ones to random values in the range `0.7 - 1.2`

## Results of Improved Model

Training for 16 epochs each of 1750 iterations where each iteration has a batch size of 64.

Results overtime:

![](readme_img/img/improved_generated_imgs.gif)

## Results of Base Model

### Loss Curve after 18000 iterations

On each iteration, the model is trained with a batch of 64 images:

![](readme_img/img/base_gan_loss_full_data.png)

### Base Model Generated Images

Each epoch ends at the completion of 1750 iterations.

#### Epoch 0

|                       Real image set                      |                    Generated image set                    |
| :-------------------------------------------------------: | :-------------------------------------------------------: |
| ![](readme_img/generated_imgs/0000_0100_real_samples.png) | ![](readme_img/generated_imgs/0000_0100_fake_samples.png) |

#### Epoch 2

|                       Real image set                      |                    Generated image set                    |
| :-------------------------------------------------------: | :-------------------------------------------------------: |
| ![](readme_img/generated_imgs/0002_0100_real_samples.png) | ![](readme_img/generated_imgs/0002_0100_fake_samples.png) |

#### Epoch 4

|                       Real image set                      |                    Generated image set                    |
| :-------------------------------------------------------: | :-------------------------------------------------------: |
| ![](readme_img/generated_imgs/0004_0100_real_samples.png) | ![](readme_img/generated_imgs/0004_0100_fake_samples.png) |

#### Epoch 6

|                       Real image set                      |                    Generated image set                    |
| :-------------------------------------------------------: | :-------------------------------------------------------: |
| ![](readme_img/generated_imgs/0006_0100_real_samples.png) | ![](readme_img/generated_imgs/0006_0100_fake_samples.png) |

#### Epoch 8

|                       Real image set                      |                    Generated image set                    |
| :-------------------------------------------------------: | :-------------------------------------------------------: |
| ![](readme_img/generated_imgs/0008_0100_real_samples.png) | ![](readme_img/generated_imgs/0008_0100_fake_samples.png) |

#### Epoch 10

|                       Real image set                      |                    Generated image set                    |
| :-------------------------------------------------------: | :-------------------------------------------------------: |
| ![](readme_img/generated_imgs/0009_1750_real_samples.png) | ![](readme_img/generated_imgs/0009_1750_fake_samples.png) |

### Test run on a tiny dataset

We trained our GAN model on the mini_data which contains images enough for only a single batch of 64 images for 100 epochs:

As expected, we can observe the general instability with training our GAN as the loss curves were not promising for training with such a small dataset:

|              GAN loss mini_data run 1             |              GAN loss mini_data run 2             |
| :-----------------------------------------------: | :-----------------------------------------------: |
| ![](readme_img/img/base_gan_loss_mini_data_1.png) | ![](readme_img/img/base_gan_loss_mini_data_2.png) |

## Acknowledgements

-   Liu, Ziwei, Ping Luo, Xiaogang Wang, and Xiaoou Tang. “Deep Learning Face Attributes in the Wild.” ArXiv:1411.7766 [Cs], September 24, 2015. <http://arxiv.org/abs/1411.7766>. **[CelebA Dataset Source]**

-   Sønderby, Casper Kaae, Jose Caballero, Lucas Theis, Wenzhe Shi, and Ferenc Huszár. “Amortised MAP Inference for Image Super-Resolution.” ArXiv:1610.04490 [Cs, Stat], February 21, 2017. <http://arxiv.org/abs/1610.04490>. **[Adding noise to images used for training the GAN]**

-   Radford, Alec, Luke Metz, and Soumith Chintala. “Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.” ArXiv:1511.06434 [Cs], January 7, 2016. <http://arxiv.org/abs/1511.06434>. **[Use of DNNs in GAN, ADAM Optimizer for Generator and SGD for Discriminator]**

-   Salimans, Tim, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. “Improved Techniques for Training GANs.” ArXiv:1606.03498 [Cs], June 10, 2016. <http://arxiv.org/abs/1606.03498>. **[One sided label smoothing]**
