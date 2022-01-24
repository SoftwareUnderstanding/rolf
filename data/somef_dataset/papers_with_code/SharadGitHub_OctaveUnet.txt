## Cityscapes-Semantic-Segmentation

This repository contains the semantic segmentation implementation on cityscapes dataset in pytorch. 

### Model

Encoder-decoder style networks, Unet (https://arxiv.org/abs/1505.04597) and Unet with octave convolution (https://arxiv.org/abs/1904.05049) are implemented. Octave convolutions (https://arxiv.org/abs/1904.05049) use high and low frequency maps to extract respective information from the image, they also have less parameters compared to vanilla convolutions. In octave convolutions, channels dimension could be controlled using alpha, which is 0.5 here, means there are equal number of channels in both low and high frquency feature maps. It was noticed that Unet with octave convolutions takes nearly 2 GB less GPU memory and give almost same performance as former. 

### Training

The network is trained using only four classes, Road, Car, sky, background. Hence masks contains only these classes. 

### Loss

For now, three different types of loss functions are used, pixelwise cross-entropy, dice coefficient and IoU loss. Since dataset has no class imbalance in images, plain cross-entropy works good.

### Output

Output masks along with actual masks for comparison are saved in "/outputs" dir with three different loss sub directories. Network with these losses is trained on different image size, hence masks vary in sizes. 
