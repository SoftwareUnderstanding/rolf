***this work is deprecated and archived by tensorflow 3 years ago because there is much better work like [HiFIC](https://github.com/edmontdants/high-fidelity-generative-compression) and alot of a good studies from google researchers and  [CLIC "Workshop and Challenge on Learned Image Compression"](http://compression.cc) which actually uses tenorflow v2.x but i created this repo because alot of people liked the results of this and preferes to try out themselves and it's a good start in this field... if you have any issues in this please let me know***

#### Full Resolution Image Compression with Recurrent Neural Networks

### Image Compression with Neural Networks
## RNN LSTM/GRU - Tensorflow Hybrid Model Residual-GRU & ResNet - Toderici2017-CVPR
# end-to-end learned based High Image Compression ratio implementation for paper Full-Resolution Lossy Image Compression CVPR17

Original | Residual GRU
:-------------------------:|:-------------------------:
![guess](Assets/example.png) | ![guess](Assets/Results%20Benchmark/example.png/png/example_15.png)

Iteration 8 without entropy coding... guess which one is the original and which is the compressed!!!

This is a [TensorFlow](http://www.tensorflow.org/) model for compressing and
decompressing images using an already trained  Residual GRU model as descibed
in [Full Resolution Image Compression with Recurrent Neural Networks](https://arxiv.org/pdf/1608.05148). Please consult the paper for more details
on the architecture and compression results.
### [Check the Pytorch version with deployed Colab](https://github.com/edmontdants/pytorch-image-comp-rnn)

This code will allow you to perform the lossy compression on an model already trained on compression. This code 
doesn't currently contain the Entropy Coding portions of our paper.

## Hardware Requirements:

* **GPU is not necessary** but preferable
* like 4Gb of RAM

## Prerequisites
The only software requirements for running the encoder and decoder is having Tensorflow installed.
You will also need to [download](https://drive.google.com/file/d/1nh4cxxds-BdsU0Tx3qP_cA1IuY2dDD5W) this compression based model [residual_gru.pb]

If you want to generate the perceptual similarity under MS-SSIM, you will also need to [Install SciPy](https://www.scipy.org/install.html).

## Encoding
The Residual GRU network is fully convolutional, but requires the images
height and width in pixels by a multiple of 32. There is an image in this folder
called example.png that is 768x1024 if one is needed for testing. We also
rely on TensorFlow's built in decoding ops, which support only PNG and JPEG at
time of release.

To encode an image, simply run the following command:

`python encoder.py --input_image=/your/image/here.png
--output_codes=output_codes.npz
--iteration=15 --model=/path/to/model/residual_gru.pb`

The iteration parameter specifies the lossy-quality to target for compression.
The quality can be [0-15], where 0 corresponds to a target of 1/8 (bits per
pixel) bpp and every increment results in an additional 1/8 bpp.

| Iteration | BPP | Compression Ratio |
|---: |---: |---: |
|0 | 0.125 | 192:1|
|1 | 0.250 | 96:1|
|2 | 0.375 | 64:1|
|3 | 0.500 | 48:1|
|4 | 0.625 | 38.4:1|
|5 | 0.750 | 32:1|
|6 | 0.875 | 27.4:1|
|7 | 1.000 | 24:1|
|8 | 1.125 | 21.3:1|
|9 | 1.250 | 19.2:1|
|10 | 1.375 | 17.4:1|
|11 | 1.500 | 16:1|
|12 | 1.625 | 14.7:1|
|13 | 1.750 | 13.7:1|
|14 | 1.875 | 12.8:1|
|15 | 2.000 | 12:1|

The output_codes file contains the numpy shape and a flattened, bit-packed
array of the codes. These can be inspected in python by using numpy.load().

[Entropy Coder](entropy%20encoder%20model/): Lossless compression of the binary representation.

## Decoding
After generating codes for an image, the lossy reconstructions for that image
can be done as follows:

`python decoder.py --input_codes=codes.npz --output_directory=/tmp/decoded/
--model=residual_gru.pb`

The output_directory will contain images decoded at each quality level.


## Comparing Similarity
One of the primary metrics for comparing how similar two images are
is MS-SSIM.

To generate these metrics on your images you can run:
`python msssim.py --original_image=/path/to/your/image.png
--compared_image=/tmp/decoded/image_15.png`

## Original Repo and Full Credits for
[google tensorflow-tensorflow](https://github.com/tensorflow/tensorflow).
[google tensorflow-compression](https://github.com/tensorflow/compression).
[google tensorflow-models](https://github.com/tensorflow/models).

[NickJohnston-google Research](https://github.com/nmjohn).
[George Toderici-google Research](https://github.com/gtoderici)

## FAQ

#### How do I train my own compression network?
currently don't provide the code to build and train a compression graph from scratch.

#### I get an InvalidArgumentError: Incompatible shapes.
This is usually due to the fact that the network only supports images that are
both height and width divisible by 32 pixel. Try padding your images to 32
pixel boundaries.


## Contact Info
Model repository maintained by Nick Johnston ([NickJohnston-google](https://github.com/nmjohn)).
