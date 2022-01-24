# Image-Segmentation-based-on-FCN
FCN implementation based on Long et al.'s paper[https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf].

Besides converting VGG-16 to FCN, I also add He et al.'s Residual Network[https://arxiv.org/abs/1512.03385] and convert Res-Net to FCN for Image Segmentation.

The final deconvolutional layer is actually transposed convolution with paddings and it is initialized with bilinear weights. For details of deconvolution and upsampling weights initialization, see this blog[https://distill.pub/2016/deconv-checkerboard/].
