# GauGanPlus
This project presents a series of modifications to NVIDIA's GauGan, https://arxiv.org/abs/1903.07291, to 
adapt better to smaller datasets. I introduce a MultiDiscriminator model combining the a single layer discriminator used in 
GauGan as well as another discriminator similar to the one used from Pix2Pix's PatchGan. The later discriminator does not 
concatenate the image  and semantic input map, it rather just uses images output from the Generator. This resulted in better 
results than using an additional L1Loss regularizer in our Generator loss function. And it was able to better adapt to unseen Data.
I also added some noise sampled from a Gaussian to the input images before being fed into the VAE Image Encoder due to the small
dataset I was working with. 

