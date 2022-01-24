# GAN-Metric
Basics to training models (to generalize to unseen data):
http://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/slides/lec12.pdf

To visualize neural network gradient descent (I watched all 4 episodes when I started learning ML):
https://www.youtube.com/watch?v=aircAruvnKk

Look at Architecture guidelines on page 3 
https://arxiv.org/abs/1511.06434

This is what we use:
https://arxiv.org/abs/1812.04948

and this:
https://arxiv.org/abs/1710.10196


TODO: 
- Autoencoder.py 
Modify Dataloader:
The filename "./annotations_slices_medium.csv" is replaced with a similar file for the anime face dataset.
This file consists of all filenames, one per line.


- Models.py
Encoder:
Modify arguments in AESConv2d so that the size of the output is half as tall, half as wide, but twice as deep;
Use filter sizes of 3x3.
EXCEPT the first hidden layer (the layer taking the image as input), which directly goes from 3 channels (RGB) to 16 channels

Bottleneck:
This pattern is repeated until the bottleneck is reached; this layer consists of 1024 or 2048 dimensions.

Decoder:
Instead of the Deconvolution operation (nn.ConvTranspose2d), we will use AESDeconv2d, which is already written.
It consists of resizing the tensor (input feature map) via interpolation, before performing regular convolutions.

