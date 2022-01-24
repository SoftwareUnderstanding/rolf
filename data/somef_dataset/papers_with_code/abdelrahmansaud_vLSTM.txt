# vLSTM

This is a PyTorch implementation of a variational LSTMCell from "A Theoretically Grounded Application of Dropout in Recurrent Neural Networks", https://arxiv.org/pdf/1512.05287

Dropout applied to each gate rather than using a whole matrix for all gates, therefor the computations are not done within a single matrix.

This doesn't work with DataParallel given PyTorch v0.4, however it should work with upcoming releases from PyTorch
