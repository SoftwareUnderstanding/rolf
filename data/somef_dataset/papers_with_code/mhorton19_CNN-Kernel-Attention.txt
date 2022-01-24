# CNN-Kernel-Attention
These are some research ideas I had to leverage global information to dynamically produce/weight CNN kernels.  These ideas are similar to squeeze-and-exitation (https://arxiv.org/pdf/1709.01507.pdf), which uses global average pooling to incorporate global information into the convolution operation.  However, where squeese and exitation uses the global average pool vector to weight the output channels, I tried to use this vector to weight/generate kernels using three different methods.  In each case, I am using a heavily augmented cifar10 dataset (rotations, shears, and flips) to advantage dynamic viewpoints, since a primary goal of these methods is to learn useful invariances/equivariances. 

kernel_weighted_cnn:

In this method, each conolution layer has a parameter containing a set of kernels as well as a parameter containing a vector corresponding to each kernel.  The global average pooling vector is reduced with a fully connected layer, and then a dot-product is performed between the reduced vector and each kernel's corresponding vector parameter.  This is then passed through a sigmoid and used to weight each convolution kernel.  This is intended to allow the supression of irrelevant filters based on viewpoint, which woud ideally allow the network to learn a rough viewpoint invariance.

TL;DR:  I am using the global information vector to weight each kernel.

fc_generated_kernel_cnn:

In this method, each convolution layer has a parameter containing a vector embedding of each kernel.  Additionally, each convolution layer contains a fully-connected network which takes the global vector and a kernel embedding as input, and outputs a kernel.  Specifically, the global vector and kernel embedding are each transformed to larger vectors, the transformed global vector is passed through a sigmoid and multiplied by the transformed kernel embedding, and then the result is transformed to a valid kernel. This gating mechanism produced superior results to appending the kernel embedding and global vector then passing through a standard feed-forward network. This method is intended to allow individual kernels to be generated using global information as well as a parameter vector.

TL;DR:  I am using the global information vector to dynamically generate each kernel.

dot_product_generated_kernel_cnn:

This method is inspired by the class of architectures which use a dot product weighted sum to attend to relevant data (transformer, neural turing machine, differentiable neural computer).  Each convolution layer has a parameter containing a set of kernels, a parameter containing a "key" vector for each kernel, a "query" vector for each input channel, and a query vector for each output channel.  To generate a vector which will be used for our dot-product weighting, we do an elementwise multiplication of each input channel query vector, output channel query vector, and reduced global content vector.  This produces a query vector which parameterizes the desired kernel between some input and output channel and accounting for global information.  A dot-product is then performed between this query vector and the key vector corresponding to each convolutional kernel.  Then, a softmax function is applied to this output vector, and the result is used to produce a weighted sum of our bank of kernels.  This architecture largely has the same goal of "fc_generated_kernel_cnn", but instead of generating each kernel using a feed-forward network, we are using dot-product attention to get each kernel as a weighted sum across a bank of kernels.  This is intended to encourage parameter sharing and promote simple kernel patterns while allowing us to dynamically produce kernels based on global information.

TL;DR:  I am using dot-product weighted summing across a small set of kernels to dynamically produce each kernel as a function of a vector associated with each input channel, output channel, and global content.
