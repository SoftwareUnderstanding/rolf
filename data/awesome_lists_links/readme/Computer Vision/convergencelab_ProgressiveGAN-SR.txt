# ProgressiveGAN-SR
Progressive GAN Super Resolution model inspired by the SRGAN and Progressive GAN from https://arxiv.org/bs/1609.04802 and https://arxiv.org/abs/1710.10196 respectively

Utlilizes:

*  Minibatch Standard Deviation
*  Fade in layers to smoothen transition between dimensions

## Normalization in Generator and Discriminator
*  Need to implement equalized learning rate
*  PixelWise Normalization after every conv 3x3 in the Generator
*  implements WGAN loss -> need to transition into GP ( Gradient Penalty )
*  Generator loss includes both critic loss and perceptual loss (VGG loss)

# Training
batch_size = 16
epochs = 256 # double for actually num iterations, as one epoch for fadein and one for straight pass
dis_per_gen_ratio = 5# number of critic trains per gen train
LAMBDA = 5# lambda for gradient penalty

## image 
UP_SAMPLE = 2 # factor for upsample
START_INPUT_DIM = 16 # start with 4x4 input -> initialize with growth phase to 8x8 (so really 4)
TARGET_DIM = 256 # full image size

## Adam 
gen_lr=0.01
dis_lr=0.01
beta_1=0.5
beta_2=0.9
epsilon=10e-8

using recommended hyper params from nvidia paper, Learning rate will likely be adjusted..
Both discriminator and Generator use adam optimizer. 
## Progressive VGG-19
Implemenets the first convolutional block of the VGG-19 for perceptual loss
weights are pre trained on ImageNet. THe VGG-19 starts with an input of 32x32 and 
grows with the GAN by factors of 2: 32->64->128->256. 

## Preprocess data
take tfds dataset, preprocess to have an lower dimensional image and higher dimensional image.
batches out dataset to batch_size.  => utilizes mapping function for tf.dataset

## fadein
For each growth period, the model must fade in the new blocks with the old configuration. 
The fadein method sets the alpha of the ProGAN to incrementally increase per each epoch in each 
growth phase. (used in weighted sum layer)

## loss functions
### discriminator
implements WGAN-GP loss, this includes the mean fake output, real output as well as
a gradient penalty. 

### generator 
Two loss functions: pre 32x32 input and post 32x32 input.
*  pre 32x32 input is WGAN loss aka -Crtic Loss. 
*  post 32x32 is WGAN loss + perceptual loss
why 32x32? This is the lowest dimension in which can incorporate VGG loss. 

## training loop
```
while input size is less than the target size:
    //fade in
    for each epoch:
        train generator one time for every five discriminator train steps. 
        update fadein 
    //stabalize
    for each epoch:
        train generator one time for every five discriminator train steps. 
    grow_Progan()
    grow_VGG()
    prepare dataset for new dims
    input size *= upscale factor
```     

### generator train step
```
    if input >= 32:
        gen loss = vgg_generator_loss
    else:
        gen loss = generator_loss

    apply gradients using Adam optimizer
```

### discriminator train step
```
    real = discriminator(realimg)
    fake = discriminator(fakeimg)
    
    dis_loss = discriminator_loss

    apply gradients using Adam optimizer
```

