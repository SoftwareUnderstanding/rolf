# DenseNet-by-Pytorch
This is my reimplement of DenseNet by Pytorch(1.1.0), which mainly refer to the [bamos](https://github.com/bamos/densenet.pytorch.git) and the paper,[*Densely Connected Convolutional Networks arXiv:1608.06993v5*](https://arxiv.org/pdf/1608.06993.pdf).
# Train and Test
`python train.py` will start the training of DensNet on the dataset *CIFAR-10*.It's worth nothing only the lastest model will be save.There are many params in `train.py`, which mentioned in the paper，such as growth_rate, optimizer and so on.Please refer to the code for more details.
# Training Loss and Test Accuracy
`python plot.py` will plot the training loss curve and test set error curve, Just like ![loss and error](/plot/sgd_cifar10_loss_error.jpg).
