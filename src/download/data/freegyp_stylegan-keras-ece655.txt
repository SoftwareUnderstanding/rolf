# stylegan-keras-ece655

![result 1076000](https://raw.githubusercontent.com/freegyp/stylegan-keras-ece655/master/image-bucket/generated_images/1076000.jpg)

![result 1074000](https://raw.githubusercontent.com/freegyp/stylegan-keras-ece655/master/image-bucket/generated_images/1074000.jpg)

We implemented the model with Tensorflow Keras based on the paper https://arxiv.org/pdf/1812.04948.pdf and trained the model on Google Cloud AI Notebook with one Nvidia Tesla v100 GPU. We used the dataset CelebA-HQ and chose the image resolution to be 256x256. The training took about 3-4 days.

The learning rate for the generator and discriminator was 0.0001 and the styler (mapping network called in the paper) 0.01 times of that.

We have saved some generated results for every 1000 global steps in the folder /image-bucket/generated_images. For each image, the results in the third row is the mix in styles of the first row and the second row in different layers. It looks like it has the trend of gradually improving on fine details. I might continue the training process when budget allows.

My personal understanding of the structure:

1. The Adaptive Instance Normalization (AdaIN) operation helps to apply weights to convolution kernels to select which kernels to use for each generator layer/block based on the styler outputs.

2. The learning rate of the styler needs to be lower for the selection of the convolution kernels to be random enough at start and fully trained. I also believe it possible that for models requiring more levels of details, it might require lower learning rates for the styler and more training time to improve on the finer details.
