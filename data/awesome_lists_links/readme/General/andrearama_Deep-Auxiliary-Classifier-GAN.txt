# Deep-Auxiliary-Classifier-GAN
High quality image generation with a modified version of Auxiliary Classifier GAN

The work is an expansion of the AC-GAN architecture introduced in the paper "Conditional Image Synthesis With Auxiliary Classifier GANs" (https://arxiv.org/pdf/1610.09585.pdf).
While in this paper the images are generated are 64x64 or 128x128, the aim of this implementation is to generate bigger images, namely 300x300.
## Curent differences with standard AC-GAN:
- Generator has an additional block formed by 'deconvolutional' (i.e. transposed convolutional) layer, Activation, Batch regularization.
- The label information is added in the Generator also after the first (just described) block. 
- Added Label smoothing for Discriminator https://arxiv.org/pdf/1606.03498.pdf
- Train the discriminator with a batch with also samples taken from a history of generated images https://arxiv.org/pdf/1612.07828.pdf (thus not only created by the current generator)
- Bilinear interpolation upsampling + normal convolutional layer instead of transposed convolutional https://distill.pub/2016/deconv-checkerboard/

## Getting Started
### Installation
- The requred libraries are opencv, numpy and keras (any recent versions would be fine).
You can install all the dependencies by:
```bash
pip install -r requirements.txt
```
- Clone this repo:
```bash
git clone https://github.com/andrearama/Deep-Auxiliary-Classifier-GAN
cd pytorch-CycleGcd Deep-Auxiliary-Classifier-GAN
```

### Train the model
Just run the train script. It is possible to change the default standards (look at the main file for more information)
```bash
python train.py 
```
### Sources:
- https://github.com/andrearama/Keras-GAN
- https://github.com/soumith/ganhacks
