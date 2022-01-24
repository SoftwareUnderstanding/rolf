# CNN-Profile-Picture
Codebase used to create a new profile picture that is the combination of a photo of me and a piece of art.

## References
### Codebase
This codebase is based on the article *Neural Style Transfer: Creating Art with Deep Learning using tf.keras and eager execution* by Raymond Yuan, published on August 3, 2018.
Link: https://medium.com/tensorflow/neural-style-transfer-creating-art-with-deep-learning-using-tf-keras-and-eager-execution-7d541ac31398

Some changes were made to the codebase in the article. The purposes of these changes are:
1. Simplification of libraries.
    * The function of libraries such as PIL or pillow were instead done using library matplotlib
2. Pythonic structuring
3. A greater number of / more descriptive comments to increase clarity.

### Model
The specific pre-trained CNN used is VGG19 (Visual Geometric Group with 19 layers). This model was first proposed in the paper "Very Deep Convolutional Networks for Large-Scale Image Recognition" by Karen Simonyan and Andrew Zisserman, published in April 2015.
Link: https://arxiv.org/pdf/1409.1556.pdf

![Example of VGG16 Structure](https://cdn-images-1.medium.com/max/1000/1*8g0VV5VKbYwawo2nfu5_qQ.png)
Caption: Example of VGG16 Structure

Photo Cite: https://medium.com/datadriveninvestor/creating-art-through-a-convolutional-neural-network-ed8a4d9a3f87