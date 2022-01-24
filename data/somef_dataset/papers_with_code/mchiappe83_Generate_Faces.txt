# Generate_Faces - Feedback and insights collected from reviews, forums & slack channel

Some Links that will help you get better understanding of GAN's:

https://arxiv.org/pdf/1511.06434.pdf
https://blog.openai.com/generative-models/
https://www.youtube.com/watch?v=YpdP_0-IEOw
https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f
Links from where you can see all variants of GAN's:

https://deephunt.in/the-gan-zoo-79597dc8c347
http://guimperarnau.com/blog/2017/03/Fantastic-GANs-and-where-to-find-them

Considerations regarding function model_inputs
TensorFlow programs use a tensor data structure to represent all data -- only tensors are passed between operations in the computation graph. You can think of a TensorFlow tensor as an n-dimensional array or list. A tensor has a static type, a rank, and a shape.

In the TensorFlow system, tensors are described by a unit of dimensionality known as rank. Tensor rank is not the same as matrix rank. Tensor rank (sometimes referred to as order or degree or n-dimension) is the number of dimensions of the tensor. For example, the following tensor (defined as a Python list) has a rank of 2:

t = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

A rank two tensor is what we typically think of as a matrix, a rank one tensor is a vector. For a rank two tensor you can access any element with the syntax t[i, j]. For a rank three tensor you would need to address an element with t[i, j, k].

http://mathworld.wolfram.com/TensorRank.html might help you with better understanding of rank in Tensor.

Considerations regarding function discriminator
https://github.com/sugyan/tf-dcgan/blob/master/dcgan.py to get an idea on implementing leaky relu.

https://github.com/sugyan/tf-dcgan/blob/master/dcgan.py to get an idea for implementing better DCGAN.

Batch normalization is recommended in DCGAN model (as mentioned in original paper). Batch normalization helps as we initialize the BatchNorm Parameters to transform the input to zero mean/unit variance distributions but during training they can learn that any other distribution might be better.
Why does this work? 
Answer Well, we know that normalization (shifting inputs to zero-mean and unit variance) is often used as a pre-processing step to make the data comparable across features. As the data flows through a deep network, the weights and parameters adjust those values, sometimes making the data too big or too small again - a problem the authors refer to as "internal covariate shift". By normalizing the data in each mini-batch, this problem is largely avoided.

http://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html to know details about Batch Normalization.

Considerations for function generator
https://github.com/soumith/ganhacks Here you can find all good tips and tricks to make GANs work.

You should be comfortable with Deconvolution, Deconvolution layer is a very unfortunate name and should rather be called a transposed convolutional layer.

https://datascience.stackexchange.com/questions/6107/what-are-deconvolutional-layers
http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf

Considerations for function model_loss
Why do we use label smoothing?
Answer: To encourage the discriminator to estimate soft probabilities rather than to extrapolate to extremely confident classification, we can use a technique called one-sided label smoothing.
http://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/

Considerations for function model_opt
When is_training parameter of batch_normalisation is true the moving_mean and moving_variance need to be updated, by default the update_ops are placed in tf.GraphKeys.UPDATE_OPS so they need to be added as a dependency to the g_train_op, thus the location chosen by you is correct. Here's a link for more details and a possible implementation.

Considerations regarding function train
* Optimize generator for more times than discriminator.

Considerations regarding Batch Size:

* If you choose a batch size too small then the gradients will become more unstable and would need to reduce the learning rate. So batch size and learning rate are linked.
* Also if one use a batch size too big then the gradients will become less noisy but it will take longer to converge.
* Recommended reading: recommend you to read http://leon.bottou.org/research/stochastic

Cosiderations regarding beta values:
* http://sebastianruder.com/optimizing-gradient-descent/index.html#adam
