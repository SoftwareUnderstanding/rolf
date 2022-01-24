# TumorDetectionDeepLearning
Using deep learning to detect tumors

The encoder-decoder temporal convolutional network model is derived from: https://arxiv.org/abs/1611.05267
Some hyperparameters are different and a flattening layer was added towards the end of the model. It is not an exact replica of the model described in the paper. Trains slowly on a GTX-1060, having taken nearly 34 minutes to train 10 epochs with a batch size of 128. Performs well on flattened MNIST data, obtaining a 98.82% training accuracy and 96.81% testing accuracy.




The dilated temporan convolutional network model is also derived from: https://arxiv.org/abs/1611.05267
This model is also not an exact replica of the one described in the paper, but it has proven to be quite lighter and much faster to train than the encoder-decoder temporal convolutional network model on the sequential MNIST data. Took 10 and a half minutes to train 15 epochs with a batch size of 128 on a GTX 1060. Performs better than the encoder-decoder T.C.N model, having obtained a 99.68% training accuracy and 97.02% testing accuracy.



The U-Net model is derived from: https://arxiv.org/abs/1505.04597
It is a close replica of the one described in the paper, with added cropping and zero padding layers to account for an odd dimension in the input shape. 

As the results show, further data preprocessing is required to obtain optimal training accuracy and prediction results. Otherwise, the model itself works as intended, showing promising results in the task of image segmentation. 

