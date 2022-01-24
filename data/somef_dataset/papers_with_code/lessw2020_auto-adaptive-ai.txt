# auto-adaptive-ai
auto adaptive framework for intrinsic hyperparameter selection, adaptive padding, normalized weights and loss standardization.

This is an adaptive neural network framework built on pytorch designed to include the following advanced features:

1 - auto-adaptive hyperparameter selection - based on https://arxiv.org/abs/1903.03088v1  (Self-Tuning Networks: Bilevel Optimization of Hyperparameters using Structured Best-Response Functions)

2 - partial convolution based padding - optimal handling of borders on images for CNN's - based on https://arxiv.org/abs/1811.11718  (Partial Convolution based Padding)

3 - weight standardization - allows for faster training with smaller batches of images - https://arxiv.org/abs/1903.10520  (Weight Standardization)

4 - standardization loss - https://arxiv.org/abs/1903.00925 (Accelerating Training of Deep Neural Networks with a Standardization Loss)



