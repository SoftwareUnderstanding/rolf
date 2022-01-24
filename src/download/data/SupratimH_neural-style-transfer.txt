# Neural Style Transfer : Generating art using Deep Learning

## Table of Contents
- [Introduction and Explanation](#programatically-painting-an-image-following-the-style-of-another-image-using-neural-style-transfer-algorithm)
- [Instruction to run this code](#instructions-to-run-this-code)
- [Dependencies](#dependencies)
- [References](#references)

## Programatically painting an image following the style of another image using Neural Style Transfer algorithm
This is an implementation of Neural Style Transfer algorithm using Tensorflow. The NST algorithm was created by Leon A. Gatys, Alexander S. Ecker and Matthias Bethge, as described in the famous paper [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576).

The core idea is to use the filter responses from different layers of a CNN (i.e. the activation values of the layers) of a convolutional network to build the style. Filter responses from different layers (ranging from lower to higher) captures from low level details (strokes, points, corners) to high level details (patterns, objects, etc) is used to modify the content image, i.e. apply the style on content image, and thus "generate" the final "painted" image.

<b>Here's an example:</b>

<b>Content/original image: ("The Milkmaid", by Raja Ravi Varma, 1904)</b>

<img src="samples/RRV-the-milkmaid.jpg" width="400px" height="600px"/>

<b>Style image: ("Self-Portrait with a Straw Hat" by Vincent van Gogh, 1887)</b>

<img src="samples/Van_Gogh-Style-400x600.jpg" width="400px" height="600px"/>

<b>Result/generated image:</b>

<img src="samples/RRV-the-milkmaid_to_VanGogh.jpg" width="400px" height="600px"/>

Usually Neural Style Transfer algorithm is applied on photograph images, with some famous painting as style, to transform the photograph into a painting-like image (sort of how Prisma app works). But being a lover of Indian and Western art, I wanted to experiment how transforming a painting of one style into another style would look like, like the example above. I have experimented with Jamini Roy's style on Raja Ravi Verma's paintings (two completely distinct styles) and the results were pretty interesting.

## Instructions to run this code:
1. Close this repository in your local system - `git clone https://github.com/SupratimH/neural-style-transfer.git` or `git clone git@github.com:SupratimH/neural-style-transfer.git`.
2. Copy your content and style images into `images` directory.
3. Make sure both are of exactly same dimensions (dim of 400 x 600 have been used in this code).
4. Update the image file names in `content_image` and `style_image` variables in `art-generation-with-nst.ipynb`.
5. Update the dimensions in `IMAGE_WIDTH` and `IMAGE_HEIGHT` variables in `nst_utils.py`.
6. Download the pre-trained VGG-19 model from http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat and save into `pretrained-model` directory.
7. Run the notebook.
8. Experiment with content and style loss hyperparamters, activation layers from which to extract style and number of epochs.

## Dependencies:
- Python3
- TensorFlow
- Scipy
- Numpy
- Matplotlib

## References:
* Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, (2015). A Neural Algorithm of Artistic Style: https://arxiv.org/abs/1508.06576
* TensorFlow Implementation of Neural Style Painting by Log0: http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style
* Karen Simonyan and Andrew Zisserman (2015). Very deep convolutional networks for large-scale image recognition: https://arxiv.org/pdf/1409.1556.pdf
* MatConvNet: http://www.vlfeat.org/matconvnet/pretrained/
* Course on "Convolution Neural Network" by https://www.deeplearning.ai/ on Coursera.
