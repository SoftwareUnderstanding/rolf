# nuclei-segmentation-unet
Nuclei segmentaion using UNET and some experiments

Implemented UNET with a few changes:
* Input has 3 channels (RGB) instead of a grayscale image
* Used transposed cpnvolutions for Upsampling
* Instead of cropping out a portion from the decoder part and concatening it in encoder part, the whole portion is so calculated to ensure the feature maps are not cropped out, rather concatenated wholly. The input image size had been adjusted accordingly. This should decrease potential loss of info due to cropping.
* Last layer is a conv2d

In addition to this:
* Loss function used is a combination of Dice loss & BCE loss

## UNET Architecture Visualization

![UNET_MODEL_VISUALIZED](https://github.com/void-in-the-matrix/nuclei-segmentation-unet/blob/main/img_model/UNET_BlockDiagram.jpg)

Refer https://arxiv.org/abs/1505.04597 for the original UNET paper!
