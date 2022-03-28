# Artistic Style Transfer Using CNN

# Decription

This model is an implementation of a research paper by Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, (2015) : A Neural Algorithm of Artistic Style.
 
Neural Style Artistic Transfer is a deep learning technique that merges two images, namely, a "content" image (A) and a "style" image (B), to create a "generated" image (G). G is a combination of A with the style of image B.

This model is implemented using Transfer Learning. It uses a previously trained convolutional network, and builds on top of that. 

Following the original paper, this model is built using the VGG19 network (19-layer version of the VGG network) 

This model has already been trained on the very large ImageNet database, and therefore has learned to recognize a variety of features at the earlier and deeper layers. This pretrained model was taken from the MatConvNet.

 # Required Modules
 
 numpy 
 
 tensorflow

os

sys

matplotlib

scipy
 
 # Building a Artistic Neural Style Transfer Algorithm
 
1.Input the content image and style image.

2.Randomly initialize the image to be generated.

3.Load the Pretrained model.

4.The content cost function is computed using one hidden layer's activations.
The style cost function for one layer is computed using the Gram matrix of that layer's activations. The overall style cost function is obtained using several hidden layers.

5.Optimizing the total cost function results in synthesizing new images.

# Sample Results

![image](https://user-images.githubusercontent.com/54103472/80914077-f97b2800-8d66-11ea-8df1-98c6919423bd.png)
![image](https://user-images.githubusercontent.com/54103472/80914096-0f88e880-8d67-11ea-854e-355acc468c77.png)

# References/Acknowledgement

1. Harish Narayanan, Convolutional neural networks for artistic style transfer. https://harishnarayanan.org/writing/artistic-style-transfer/

2. MatConvNet. http://www.vlfeat.org/matconvnet/pretrained/

3. Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, (2015). A Neural Algorithm of Artistic Style (https://arxiv.org/abs/1508.06576)

4. Matthew D, Rob Fergus (2013), Visualizing and Understanding Convolutional Networks. https://arxiv.org/pdf/1311.2901.pdf
