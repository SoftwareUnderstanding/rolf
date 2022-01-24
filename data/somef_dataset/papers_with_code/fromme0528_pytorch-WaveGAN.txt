# pytorch-WaveGAN (Not Yet Completed)
A pytorch implementation of WaveGAN 
[(Donahue, et al. 2018)](https://arxiv.org/abs/1802.04208).

WaveGAN is first approach to synthesize raw audio using GAN.

# Features
* Overall architecture is based on DCGAN
* 2D Conv(5,5) -> 1D Conv(1,25)
* original DCGAN output size is 4096. Add one layer to make output size larger(16384).
* 16384 is slightly more than 1 second raw audio of 16kHz
* change audio data 16-bit to 32=bit floating point
* Train a post-processing filter to aviod checkerboard effects
* Phase shuffle to avoid for Discriminator not to train checkerboard effects