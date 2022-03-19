# DeepSlowMotion
A deep convolutional neural network for multi-frame video interpolation. Based on the work of Huaizu Jiang, Deqing Sun, Varun Jampani, Ming-Hsuan Yang, Erik Learned-Miller and Jan Kautz. To see the original work, please see:

>H. Jiang, D. Sun, V. Jampani, M.-H. Yang, E. Learned-Miller and J. Kautz, "Super SloMo: High Quality Estimation of Multiple Intermediate Frames for Video Interpolation," _Proceedings of the The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_, Salt Lake City, UT, USA, 2018, pp. 9000-9008.

or go to: [arXiv:1712.00080](https://arxiv.org/abs/1712.00080)

The script `download_dataset.sh` downloads two datasets, the Adobe 240-fps dataset and the Need for Speed dataset, that can be used for training the neural network. For more information on these datasets, please see:

>S. Su, M. Delbracio, J. Wang, G. Sapiro, W. Heidrich and O. Wang, "Deep Video Deblurring for Hand-Held Cameras," _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_, Honolulu, HI, USA, 2017, pp. 237-246.

or go to: [arXiv:1611.08387](https://arxiv.org/abs/1611.08387)

> H. K. Galoogahi, A. Fagg, C. Huang, D. Ramanan and Simon Lucey, "Need for Speed: A Benchmark for Higher Frame Rate Object Tracking," _Proceedings of the IEEE International Conference on Computer Vision (ICCV)_, Venice, Italy, 2017, pp. 1125-1134.

or go to: [arXiv:1703.05884](https://arxiv.org/abs/1703.05884)

In order to download both datasets with the script, just type the following command in a terminal:

```bash
./download_dataset.sh path_to_folder
```

where `path_to_folder` is the directory where the dataset should be downloaded. This directory is created if it does not exist.

The loss function includes a term that depends on the activations of a convolutional layer of the VGG16 neural network. The pretrained  weights of the VGG16 neural networks are stored in the `vgg16_weights_no_fc.npz` file. This file is a modified version of the file that can be found in [this link](https://www.cs.toronto.edu/~frossard/post/vgg16/). In order to reduce the size of the file, the weights of the fully connected layers have been removed. For more information on the VGG16 neural network, please see:

>K. Simonyan and A. Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition," Proceedings of the International Conference on Learning Representations (ICLR), San Diego, CA, USA, 2015, pp. 1-14.

or go to: [arXiv:1409.1556](https://arxiv.org/pdf/1409.1556.pdf)
