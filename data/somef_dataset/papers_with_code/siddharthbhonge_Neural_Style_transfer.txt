# What if Statue of Liberty was painted by Picasso?

Neural Style learning is one of the most exciting sides of deep learning.Ever wondered if Angelina Jolie was painted by Da Vinci?<br />

![alt text](https://github.com/siddharthbhonge/Neural_Style_transfer/blob/master/demo1.jpg)

![alt text](https://github.com/siddharthbhonge/Neural_Style_transfer/blob/master/demo2.jpg)

## Installation

 - Keras
 - TensorFlow
 - Scipy
 - Numpy

## Details

  #### What is actually happening?
  
  Neural Style Transfer (NST) uses a previously trained convolutional network, and builds on top of that. The idea of using a network trained on a different task and applying it to a new task is called transfer learning.  <br />

 Neural Style Transfer is an algorithm that given a content image C and a style image S can generate an artistic image  <br />
 It uses representations (hidden layer activations) based on a pretrained ConvNet.
 The content cost function is computed using one hidden layer's activations.<br />
 The style cost function for one layer is computed using the Gram matrix of that layer's activations. <br />
 The overall style cost function is obtained using several hidden layers.<br/>
 Optimizing the total cost function results in synthesizing new images.<br />
  
  

 ####  Content Cost
![alt text](https://github.com/siddharthbhonge/Neural_Style_transfer/blob/master/content_cost.jpg)

  



 #### Style Cost

![alt text](https://github.com/siddharthbhonge/Neural_Style_transfer/blob/master/style_cost.jpg)

 #### Total Cost

![alt text](https://github.com/siddharthbhonge/Neural_Style_transfer/blob/master/total_cost.png)


## Note

* The style of an image can be represented using the Gram matrix of a hidden layer's activations. However, we get even better results combining this representation from multiple different layers.<br /> 
* This is in contrast to the content representation, where usually using just a single hidden layer is sufficient.<br />
*Minimizing the style cost will cause the image GG to follow the style of the image SS. <br />
 
* Please Download VGG19 weights and keep in /model folder.https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d
## Acknowledgemnts 

*Siddharth Bhonge https://github.com/siddharthbhonge 




## Reference

Andrew Ng's Deep Learning Specialization.<br />


 Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, (2015). A Neural Algorithm of Artistic Style (https://arxiv.org/abs/1508.06576)<br />
 Harish Narayanan, Convolutional neural networks for artistic style transfer. https://harishnarayanan.org/writing/artistic-style-transfer/<br />
 Log0, TensorFlow Implementation of "A Neural Algorithm of Artistic Style". http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style<br />
 Karen Simonyan and Andrew Zisserman (2015). Very deep convolutional networks for large-scale image recognition (https://arxiv.org/pdf/1409.1556.pdf)<br />
 MatConvNet. http://www.vlfeat.org/matconvnet/pretrained/<br />


