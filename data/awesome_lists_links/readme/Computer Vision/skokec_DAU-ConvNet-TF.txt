# DAU-ConvNet in pure TensorFlow

Python/TensorFlow implementation of the Displaced Aggregation Units for Convolutional Networks from CVPR 2018 paper titled [*"Spatially-Adaptive Filter Units for Deep Neural Networks"*](http://openaccess.thecvf.com/content_cvpr_2018/papers/Tabernik_Spatially-Adaptive_Filter_Units_CVPR_2018_paper.pdf) that was developed as part of [Deep Compositional Networks](http://www.vicos.si/Research/DeepCompositionalNet).

This code is a less efficient version of [DAU-ConvNet](http://github.com/skokec/DAU-ConvNet) that is implemented using only Python/TensorFlow operations and results in:

 * fully learnable sigma/standard deviation of DAU (independently for each DAU as done in our [ICPR 2016 paper](https://prints.vicos.si/publications/339)),
 * having automatically differentiable operations using the auto-grad in TensorFlow, 
 * is suitable for prototyping due to its flexibility, and
 * is easy to use (only Python code)
 
however, it comes with a few disadvantages: 

 * performance is dependent on the kernel size based on max displacements,
 * is slightly slower code for large displacements, and
 * uses more GPU memory.

## Citation ##
Please cite our CVPR 2018 paper when using the DAU code/model:

```
@inproceedings{Tabernik2018,
	title = {{Spatially-Adaptive Filter Units for Deep Neural Networks}},
	author = {Tabernik, Domen and Kristan, Matej and Leonardis, Ale{\v{s}}},
	booktitle = {2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition},
	year = {2018}
	pages = {9388--9396}
}
```
## Usage 

Basically this implementation converts DAUs into a single K x K kernel, and then uses a standard conv2d operation, which can be simply done by:

```Python
    
    F = 128 # num output channels
    G = 4 # num DAUs per channel
    S = 64 # num input channels    
    max_kernel_size = 17
    
    dau_w = tf.Variable(shape=(1,S,G,F))
    dau_mu1 = tf.Variable(shape=(1,S,G,F))
    dau_mu2 = tf.Variable(shape=(1,S,G,F))
    dau_sigma = tf.Variable(shape=(1,S,G,F)) 
    
    [X,Y] = np.meshgrid(np.arange(max_kernel_size),np.arange(max_kernel_size))
    
    X = np.reshape(X,(max_kernel_size*max_kernel_size,1,1,1)) - int(max_kernel_size/2)
    Y = np.reshape(Y,(max_kernel_size*max_kernel_size,1,1,1)) - int(max_kernel_size/2)
        
    # Gaussian kernel
    gauss_kernel = tf.exp(-1* (tf.pow(X - dau_mu1,2.0) + tf.pow(Y - dau_mu2,2.0)) / (2.0*tf.pow(dau_sigma,2.0)),name='gauss_kernel')
    gauss_kernel_sum = tf.reduce_sum(gauss_kernel,axis=0, keep_dims=True,name='guass_kernel_sum')
    gauss_kernel_norm = tf.divide(gauss_kernel, gauss_kernel_sum ,name='gauss_kernel_norm')
    
    # normalize to sum of 1 and add weight
    gauss_kernel_norm = tf.multiply(dau_w, gauss_kernel_norm,name='gauss_kernel_weight')
    
    # sum over Gaussian units
    gauss_kernel_norm = tf.reduce_sum(gauss_kernel_norm, axis=2, keep_dims=True,name='gauss_kernel_sum_units')
    
    # convert to [Kw,Kh,S,F] shape
    gauss_kernel_norm = tf.reshape(gauss_kernel_norm, (max_kernel_size, max_kernel_size, gauss_kernel_norm.shape[1], gauss_kernel_norm.shape[3]),name='gauss_kernel_reshape')
      
    output = tf.nn.conv2d(inputs, gauss_kernel_norm)
    
```

## Usage with tf.contrib.layer.conv2d compatible API

We provide a wrapper based on `tf.contrib.layer.conv2d()` API, that is also compatible/interchangeable with the `dau_conv2d` from [DAU-ConvNet](http://github.com/skokec/DAU-ConvNet). 

Install using pip:
```bash
sudo pip3 install https://github.com/skokec/DAU-ConvNet-TF/releases/download/v1.0/dau_conv_tf-1.0-py3-none-any.whl  
```

There are two available methods to use: 

```python
from dau_conv_tf import dau_conv2d_tf

dau_conv2d_tf(inputs,
             filters, # number of output filters
             dau_units, # number of DAU units per image axis, e.g, (2,2) for 4 DAUs per filter 
             max_kernel_size, # maximal possible size of kernel that limits the offset of DAUs (highest value that can be used=17)  
             stride=1, # only stride=1 supported 
             mu_learning_rate_factor=500, # additional factor for gradients of mu1 and mu2
             dau_unit_border_bound=1,
             data_format=None,
             activation_fn=tf.nn.relu,
             normalizer_fn=None,
             normalizer_params=None,
             weights_initializer=tf.random_normal_initializer(stddev=0.1), 
             weights_regularizer=None,
             mu1_initializer=None, 
             mu1_regularizer=None, 
             mu2_initializer=None,
             mu2_regularizer=None,
             sigma_initializer=None,
             sigma_regularizer=None,
             biases_initializer=tf.zeros_initializer(),
             biases_regularizer=None,
             reuse=None,
             variables_collections=None,
             outputs_collections=None,
             trainable=True,
             scope=None)
```
 
```python
from dau_conv_tf import DAUConv2dTF

DAUConv2dTF(filters, # number of output filters
           dau_units, # number of DAU units per image axis, e.g, (2,2) for 4 DAUs total per one filter
           max_kernel_size, # maximal possible size of kernel that limits the offset of DAUs (highest value that can be used=17)
           strides=1, # only stride=1 supported
           data_format='channels_first', # supports only 'channels_last' 
           activation=None,
           use_bias=True,
           weight_initializer=tf.random_normal_initializer(stddev=0.1),
           mu1_initializer=None, 
           mu2_initializer=None, 
           sigma_initializer=None,
           bias_initializer=tf.zeros_initializer(),
           weight_regularizer=None,
           mu1_regularizer=None,
           mu2_regularizer=None,
           sigma_regularizer=None,
           bias_regularizer=None,
           activity_regularizer=None,
           weight_constraint=None,
           mu1_constraint=None,
           mu2_constraint=None,
           sigma_constraint=None,
           bias_constraint=None,
           trainable=True,
           mu_learning_rate_factor=500, # additional factor for gradients of mu1 and mu2
           dau_unit_border_bound=1,  
           unit_testing=False, # for competability between CPU and GPU version (where gradients of last edge need to be ignored) during unit testing
           name=None)
```

