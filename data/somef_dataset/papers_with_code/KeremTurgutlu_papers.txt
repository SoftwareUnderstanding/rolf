# Summarize Deep Learning Papers

Skips math part, focuses on why are researchers solve the particular problem, what are the results and so on.

## Papers 

### 1) Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network

- Link: https://arxiv.org/pdf/1609.05158.pdf

- Motivation: Finding an efficient way of upscaling from low resolution to high resolution.

- Proposition: Fractional $\frac{1}{r}$ convolution is implemented as Pixel Shuffle. PS rearranges elements of a tensor from HxWxC*r^2 to rHxrWxC.

- Results: Better PSNR (dB) - MSE like metric for image resolution, less parameters, faster training and inference.
