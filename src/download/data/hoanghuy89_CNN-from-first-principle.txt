# CNN from scratch usinng only Numpy for MNIST and CIFAR10
This notebook implement a mini deep learning frame-work in the style of pytorch. Everything here from RELU, Batch Norm, Softmax, ...  are implemented from scratch, in an attemp to fully understand how Convolutional neuron netowrk works under the hood.

<img src="images/yes-no.png" style="width:600px;height:300px;">

Collab link: https://colab.research.google.com/github/hoanghuy89/CNN-from-first-principle/blob/main/CNN-from-first-principle.ipynb

The purpose of this notebook is to encourage understanding of deep learning, everything under the hood that enable these stuffs to work (which is not scary at all) and the frameworks that implement them. A little bit of programming skill and linnear algebra is sufficient to work your way from bottom up to implement convolution neural networks. I also include a graident check function that helps validate back-probagation implement.

The list of layers and activation that were implemented from scratch in this repo:

- Convolution net for image (http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
- Pooling (max and average) 
- Batch normalization (https://arxiv.org/abs/1502.03167)
- RELU-Rectified Linear Unit (https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf) 
- Linear
- Softmax
- Adam optimization (https://arxiv.org/abs/1412.6980)
- Cross entropy loss and Mean square error

MNIST and CIFAR10 datsets are used for evaluation.

The network declaration and using is similar to that of pytorch. 
```python
class CNN:
    """
    Typical pytorch style network declaration.
    """
    def __init__(self, in_shape, out_size):

        in_size = in_shape[0]
        in_chn = in_shape[-1]

        
        self.convo1 = Conv2D(in_chn=in_chn, out_chn=12, kernel_size=3, in_shape=in_shape, padding=1, stride=2, bias=False)

        w_x = h_x = (in_size-3 + 2*1)//2 + 1
        output_shape = (w_x,h_x,12)

        self.batchnorm1 = BatchNorm(output_shape)

        
        self.convo2 = Conv2D(in_chn=12, out_chn=24, kernel_size=3, in_shape=output_shape, padding=1, stride=2, bias=False)

        w_x = h_x = (w_x-3 + 2*1)//2 + 1
        output_shape = (w_x,h_x,24)

        self.batchnorm2 = BatchNorm(output_shape)

        self.flatten = Flatten()


        linear_in = np.prod(output_shape)

        self.linear_softmax = Linear_SoftMax(linear_in, out_size)

        # only layers with trainable weights here, which are used in optimization/gradient update.
        self.layers = {'convo1': self.convo1, 'batch_norm1': self.batchnorm1, 'convo2': self.convo2, 
                       "batch_norm2": self.batchnorm2, 'linear_softmax': self.linear_softmax}

    def forward(self, X):

        X = self.convo1.forward(X)
        X = self.batchnorm1.forward(X)

        X = self.convo2.forward(X)
        X = self.batchnorm2.forward(X)

        X = self.flatten.forward(X)
        X = self.linear_softmax.forward(X)

        return X

    def backward(self, dZ):

        dZ = self.linear_softmax.backward(dZ)
        dZ = self.flatten.backward(dZ)

        dZ = self.batchnorm2.backward(dZ)
        dZ = self.convo2.backward(dZ)

        dZ = self.batchnorm1.backward(dZ)
        dZ = self.convo1.backward(dZ)

        return dZ

    def set_weights(self, weight_list):
        for k, (W,b) in weight_list.items():
            self.layers[k].W = W
            self.layers[k].b = b

    def get_weights(self):
        return {k:(layer.W, layer.b) for k,layer in self.layers.items()}

    def get_dweights(self):
        return {k:(layer.dW, layer.db) for k,layer in self.layers.items()}
```
