# PyTorch-CoordConv
Unofficial CoordConv nn.Module implementation and using it for supervised regression on the Not-so_Clevr dataset.
For more information, check out [Uber's blog](https://eng.uber.com/coordconv/) and [paper on arxiv](https://arxiv.org/pdf/1807.03247.pdf).

## Generating the Not-so-Clevr dataset in PyTorch

As described in the research paper, the Not-so-Clevr dataset can be generated in 2 lines of code with tensorflow and numpy as show below:
```
onehots = np.pad(np.eye(3136).reshape((3136, 56, 56, 1)), ((0,0), (4,4), (4,4), (0,0)), "constant");
images = tf.nn.conv2d(onehots, np.ones((9, 9, 1, 1)), [1]*4, "SAME")
```
I replicated the data generation in PyTorch, which requires more lines but is likewise just as simple, as such:
```
  n_samples = 56 ** 2
  onehots = np.pad(np.eye(n_samples).reshape((n_samples, 1, 56, 56)), ((0,0), (0, 0), (4,4), (4,4)), "constant")
  onehots = torch.from_numpy(onehots).float()
  with torch.no_grad():
      conv = nn.Conv2d(1, 1, kernel_size=9, padding=4, stride=1)
      conv.weight.data.fill_(1)
      conv.bias.data.fill_(0)
      dataset_x = conv(onehots)
```
## CoordConv

Convolution is an equivariant operation, meaning that the model doesn't have a means to know *where* in a feature map that filters are activating. CoordConv fixes this problem by concatenating two channels that add spatial information to images.

From Uber's AI Lab:

![](http://eng.uber.com/wp-content/uploads/2018/07/image8.jpg)

In cases where equivariance is actually helpful, CoordConv layers still don't hurt model performance, because filter weights over the spatial channels can become zero.

## Performance

In the supervised regression task, the input is a 1-dimensional image array, and the model trains to predict the coordinates of the top-left corner of the square in the image.

Example input:

![](https://github.com/gnouhp/PyTorch-CoordConv/blob/master/repoimages/sampleinput.png)


In multiple attempts, I wasn't able to get significantly better performance with a small network with a CoordConv module as the first layer. In the papers, the authors use a network with a few more layers and a global pooling layer instead of the fully-connected layers I use, so I would be interested in whether or not the CoordConv and global pooling layers have a unique relationship.

Performance graph:

![](https://github.com/gnouhp/PyTorch-CoordConv/blob/master/repoimages/resultsplot.png)

I may attempt to exactly duplicate the model architecture and hyperparameters exactly in the future and see if I can replicate the performance. I'll have to learn more about global pooling and how to implement it before I replace the fully-connected layers with it.
