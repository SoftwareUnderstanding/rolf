# Spatial pyramid pooling

Module for the spatial pyramid pooling (SPP) module used in classification, object detection, and segmentation tasks in
computer vision. In a general sense, along with techniques like dilated convolutions and other feature pyramid methods,
SPP is said to endow local predictions with regional and global context. For example, in the segmentation case, the
final prediction is made using a stack of feature maps, one globally pooled (1x1), one pooled into 4 regions (2x2), etc.
These are stacked with the regular CNN base's feature maps for final convolution into class logits for prediction. In
this way, each pixel's prediction has direct access to context about the entire image, and its region, as well as local
features. So, a pixel might be more likely to be considered part of a car if it sees that road-like textures are in the
image, or that a tree is nearby.

This module is written to accommodate two styles of feature concatenation:

1. Flattened (1D) concatenation, a la https://arxiv.org/abs/1406.4729

    Here, the resulting feature maps are flattened into (batch, -1) tensors, and then concatenated along dim 1. This
    results in a (batch, sum_i (l_i * l_i * in_channels)) output tensor. This more or less faithfully recreates the SPP
    module in the original spatial pyramid pooling paper.
2. Feature map (2D) concatenation, a la https://arxiv.org/abs/1612.01105

    Here, the resulting feature maps are not flattened, but rather concatenated into (batch, C', H, W) tensors, where C'
    is the sum of the channels in the incoming feature maps, plus all pooled feature maps. This results in a
    (batch, 2 * in_channels - in_channels % levels, H, W) output tensor, where the in_channels % levels difference is
    due to floor division in computing the number of channels each pooled feature map should have, when in_channels is
    not divisible by the number of levels. This more or less faithfully recreates the SPP module in the PSPNet paper.

Note that, in the first case, the function would have no weights, and could therefore doesn't need to be represented as
a Module, but in the second case, convolutions with weights are needed.

This code was written and commented for minimalism and ease of understanding. Yes, it would be easier to do this with
one of the Adaptive pooling functions, but I find that the mechanics of those functions are less visible than if we
stick with doing the pooling size computations explicitly.