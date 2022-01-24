## About

We optimize operation `CoordConv` from the paper https://arxiv.org/pdf/1807.03247.pdf.
For this purpose we use simple math:
- convolution for input with `c1 + c2` channels can be expressed as the sum of convolutions for input with `c1` and `c2` channels separately.
- for channel matrix which has equal rows `(0, 1, .. n)` or equal columns `(0, 1, .. n)^T`
(these matrices are added in the CoordConv defining the coordinates) convolution operation equals to
 - matrix with columns `alpha * (0, 1, .., n)^T + alpha0`
 - matrix with rows `beta * (0, 1, .., n) + beta0`
correspondingly. Bias can be ignored due to bias in the convolution layer which comes before this operation.

In the `coordconv/modules` a layer `CoordXY` is provided which adds necessary result to the standard convolution to obtain `CoordConv`.
With comparison to the original paper, optimized version of operation is simpler, more efficient, and has less number of parameters.

## Dependencies
- python 3.6
- pytorch 0.4