# Class-Conditional Image Generation using Projection Discriminators

An un-official PyTorch implementation for the paper *cGAN with Projection Discriminator*, arXiv: 1802.05637.

The author's implementation using Chainer can be found in this repo [here] (https://github.com/pfnet-research/sngan_projection)

## Techniques in cGAN

The cGAN network build on top of the Spectral Normalization method, which is an effective way to ensure K-Lipschitz continuity at each layer and thus bounds the gradient of the discriminator. As proposed in the [paper](https://arxiv.org/abs/1802.05637), we fixed the spectral norm of the layer by replacing the original weight *w* with *W/σ(W)*, where σ(W) is the largest singular value of w. This significantly improves the stability of the training, and made the multi-class generative network possible.

It is implemented in the Discriminator's residual blocks using PyTorch's *torch.nn.utils.spectral_norm*
'''
    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1,
                 activation=F.relu, downsample=False):
        super(Block, self).__init__()

        self.activation = activation
        self.downsample = downsample

        self.learnable_sc = (in_ch != out_ch) or downsample
        if h_ch is None:
            h_ch = in_ch
        else:
            h_ch = out_ch

        self.c1 = utils.spectral_norm(nn.Conv2d(in_ch, h_ch, ksize, 1, pad))
        self.c2 = utils.spectral_norm(nn.Conv2d(h_ch, out_ch, ksize, 1, pad))
        if self.learnable_sc:
            self.c_sc = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

        self._initialize()
'''

The cGAN paper introduced a new way to add conditioning to the network. While earlier works on conditional GANs explored ways to concatenate the class-encoding vector *y* into part of the network, the authors of *cGAN with Projection Discriminators* shows that by incorporating *y* as an inner product with the network, it can more effectively interact with the hidden layers of the network and allow for more stable training.

![Comparisons of different conditioning for the discriminator](images/cgan_paper_fig1.png)

To implement class-conditional batch normalization, we first need to create a base class for conditional batch normalization. This should both inherit from PyTorch's BatchNorm2d class, and introduce the conditional weighting and bias terms for the batch normalization layer:
'''
class ConditionalBatchNorm2d(nn.BatchNorm2d):

    """Conditional Batch Normalization"""

    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(ConditionalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, input, weight, bias, **kwargs):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        output = F.batch_norm(input, self.running_mean, self.running_var,
                              self.weight, self.bias,
                              self.training or not self.track_running_stats,
                              exponential_average_factor, self.eps)
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)
        size = output.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * output + bias 
'''

On top of this, we create the CategoricalConditionalBatchNorm2d that extends the weight and bias to accommodate for any number of classes. By using *torch.nn.Embedding*, we turn the weights and biases to a look up table that correspond to each class label:
'''
class CategoricalConditionalBatchNorm2d(ConditionalBatchNorm2d):

    def __init__(self, num_classes, num_features, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(CategoricalConditionalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.weights = nn.Embedding(num_classes, num_features)
        self.biases = nn.Embedding(num_classes, num_features)

        self._initialize()

    def _initialize(self):
        init.ones_(self.weights.weight.data)
        init.zeros_(self.biases.weight.data)

    def forward(self, input, c, **kwargs):
        weight = self.weights(c)
        bias = self.biases(c)

        return super(CategoricalConditionalBatchNorm2d, self).forward(
                     input, weight, bias)
'''

## Output Results

For training, I used the [Tiny ImageNet](https://tiny-imagenet.herokuapp.com/) dataset, which by default contains 64*64 images from 200 categories.

I first trained using only the spectrial normalization GAN, without conditioning on class labels. While the result after 40000 iterations shows improvements in capturing edges and textures, it is far from synthesizing high level features.

![output1_sngan](images/output1_sngan.png)

I then used cGAN to train for the same dataset. However, the output images after the same number of interations didn't show better class-based features than the spectrial normalization GAN. 
My interpretation is that the total number of classes for a GAN can not be arbitrarily large. The number of categories to train for should depend on the network's input feature size.

![output0](images/output0.png)

Eventually, I trained with a smaller number of image classes, and was able to get class-specific generative results. 

![output1](images/output1.jpg)

I also noticed that the training is significantly faster when the training images have simpler geometries and textures. For example, to train for 6 categories of geometric shapes only took 6000 interations to get a good result. his allows me to quickly test out feature interpolation.

![colors_1](images/colors_1.jpg) ![interpolate](images/interpolate.jpg)

Here is a [link](https://colab.research.google.com/drive/1HLZBceHtiz_aTjNw_QP1yGB7HFJYePAv) to Colab Demo for using the Generator and mixing class features for the output image.
