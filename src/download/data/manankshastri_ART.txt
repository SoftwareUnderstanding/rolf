# Art Generation using Neural Style Transfer

Neural Style Transfer - algorithm created by Gatys et al. (2015) (https://arxiv.org/abs/1508.06576). 

**Here we will**:
- Implement the neural style transfer algorithm
- Generate novel artistic images using the algorithm

## 1. Problem Statement
Neural Style Transfer (NST) is one of the most fun techniques in deep learning. As seen below, it merges two images, namely, a "content" image (C) and a "style" image (S), to create a "generated" image (G). The generated image G combines the "content" of the image C with the "style" of image S. 

For example, an image of the Louvre museum in Paris (content image C), mixed with a painting by Claude Monet, a leader of the impressionist movement (style image S).

<img src="images/louvre_generated.png" style="width:750px;height:200px;">

## 2. Transfer Learning
Neural Style Transfer (NST) uses a previously trained convolutional network, and builds on top of that. The idea of using a network trained on a different task and applying it to a new task is called transfer learning. 

Following the original NST paper (https://arxiv.org/abs/1508.06576), we will use the VGG network. Specifically, we'll use VGG-19, a 19-layer version of the VGG network. This model has already been trained on the very large ImageNet database, and thus has learned to recognize a variety of low level features (at the earlier layers) and high level features (at the deeper layers).

The model is stored in a python dictionary where each variable name is the key and the corresponding value is a tensor containing that variable's value. To run an image through this network, we just have to feed the image to the model. In TensorFlow, we can do so using the [tf.assign](https://www.tensorflow.org/api_docs/python/tf/assign) function. In particular, we will use the assign function like this:  
```python
model["input"].assign(image)
```
This assigns the image as an input to the model. After this, if we want to access the activations of a particular layer, say layer `4_2` when the network is run on this image, we would run a TensorFlow session on the correct tensor `conv4_2`, as follows:  
```python
sess.run(model["conv4_2"])
```

## 3 - Neural Style Transfer 

We will build the NST algorithm in three steps:

- Build the content cost function **J<sub>content</sub>(C,G)**
- Build the style cost function **J<sub>style</sub>(S,G)**
- Put it together to get **J(G) = &#945; J<sub>content</sub>(C,G) + &#946; J<sub>style</sub>(S,G)**

### 3.1 - Computing the content cost
### 3.1.1 - How do you ensure the generated image G matches the content of the image C?

The earlier (shallower) layers of a ConvNet tend to detect lower-level features such as edges and simple textures, and the later (deeper) layers tend to detect higher-level features such as more complex textures as well as object classes. 

We would like the "generated" image G to have similar content as the input image C. Suppose we have chosen some layer's activations to represent the content of an image. In practice, we'll get the most visually pleasing results if we choose a layer in the middle of the network--neither too shallow nor too deep. 

So, suppose we have picked one particular hidden layer to use. Now, set the image C as the input to the pretrained VGG network, and run forward propagation. Let **a<sup>(C)</sup>** be the hidden layer activations in the layer you had chosen. This will be a **n_H x n_W x n_C** tensor. Repeat this process with the image G: Set G as the input, and run forward progation. Let **a<sup>(G)</sup>** be the corresponding hidden layer activation. We will define as the content cost function as:

<img src="images/eq1.PNG" style="width:800px;height:400px;">

Here, **n_H, n_W** and **n_C** are the height, width and number of channels of the hidden layer we have chosen, and appear in a normalization term in the cost. For clarity, note that **a<sup>(C)</sup>** and **a<sup>(G)</sup>** are the volumes corresponding to a hidden layer's activations. In order to compute the cost **J<sub>content</sub>(C,G)**, it might also be convenient to unroll these 3D volumes into a 2D matrix, as shown below. (Technically this unrolling step isn't needed to compute **J<sub>content</sub>**, but it will be good practice for when you do need to carry out a similar operation later for computing the style const **J<sub>style</sub>**.)

<img src="images/NST_LOSS.png" style="width:800px;height:400px;">


### 3.2 - Computing the style cost
### 3.2.1 - Style matrix

The style matrix is also called a "Gram matrix." In linear algebra, the Gram matrix G of a set of vectors **(v<sub>1</sub>,... ,v<sub>n</sub>)** is the matrix of dot products, whose entries are <img src="images/eq2.PNG" style="width:750px;height:200px;">. In other words, **G<sub>ij</sub>** compares how similar **v_i** is to **v_j**: If they are highly similar, you would expect them to have a large dot product, and thus for **G<sub>ij</sub>** to be large. 

In NST, you can compute the Style matrix by multiplying the "unrolled" filter matrix with their transpose:

<img src="images/NST_GM.png" style="width:900px;height:300px;">

The result is a matrix of dimension **(n<sub>C</sub>,n<sub>C</sub>)** where **n<sub>C</sub>** is the number of filters. The value **G<sub>ij</sub>** measures how similar the activations of filter **i** are to the activations of filter **j**. 

One important part of the gram matrix is that the diagonal elements such as **G<sub>ii</sub>** also measures how active filter **i** is. For example, suppose filter **i** is detecting vertical textures in the image. Then **G<sub>ii</sub>** measures how common vertical textures are in the image as a whole: If **G<sub>ii</sub>** is large, this means that the image has a lot of vertical texture. 

By capturing the prevalence of different types of features (**G<sub>ii</sub>**), as well as how much different features occur together (**G<sub>ij</sub>**), the Style matrix **G** measures the style of an image. 

### 3.2.2 - Style cost
After generating the Style matrix (Gram matrix), your goal will be to minimize the distance between the Gram matrix of the "style" image S and that of the "generated" image G. For now, we are using only a single hidden layer **a<sup>[l]</sup>**, and the corresponding style cost for this layer is defined as: 

<img src="images/eq3.PNG" style="width:750px;height:200px;">

where **G<sup>(S)</sup>** and **G<sup>(G)</sup>** are respectively the Gram matrices of the "style" image and the "generated" image, computed using the hidden layer activations for a particular hidden layer in the network.  

### 3.2.3 - Style Weights

So far we have captured the style from only one layer. We'll get better results if we "merge" style costs from several different layers. Feel free to experiment with different weights to see how it changes the generated image **G**. 

You can combine the style costs for different layers as follows:
<img src="images/eq4.PNG" style="width:750px;height:200px;">

### 3.3 - Defining the total cost to optimize
Finally, let's create a cost function that minimizes both the style and the content cost. The formula is: 

**J(G) = &#945; J<sub>content</sub>(C,G) + &#946; J<sub>style</sub>(S,G)**


## 4 - Solving the optimization problem
Finally, let's put everything together to implement Neural Style Transfer!

1. Create an Interactive Session
2. Load the content image 
3. Load the style image
4. Randomly initialize the image to be generated 
5. Load the VGG19 model
7. Build the TensorFlow graph:
    - Run the content image through the VGG19 model and compute the content cost
    - Run the style image through the VGG19 model and compute the style cost
    - Compute the total cost
    - Define the optimizer and the learning rate
8. Initialize the TensorFlow graph and run it for a large number of iterations, updating the generated image at every step.

## Things to remember
- The content cost takes a hidden layer activation of the neural network, and measures how different **a<sup>(C)</sup>** and **a<sup>(G)</sup>** are. 
- When we minimize the content cost later, this will help make sure **G** has similar content as **C**.
- The style of an image can be represented using the Gram matrix of a hidden layer's activations. However, we get even better results combining this representation from multiple different layers. This is in contrast to the content representation, where usually using just a single hidden layer is sufficient.
- Minimizing the style cost will cause the image **G** to follow the style of the image **S**. 
- The total cost is a linear combination of the content cost **J<sub>content</sub>(C,G)** and the style cost **J<sub>style</sub>(C,G)**
- **&#945;** and **&#946;** are hyperparameters that control the relative weighting between content and style.

***
Here are few examples:

- The beautiful ruins of the ancient city of Persepolis (Iran) with the style of Van Gogh (The Starry Night)
<img src="images/perspolis_vangogh.png" style="width:750px;height:300px;">

- The tomb of Cyrus the great in Pasargadae with the style of a Ceramic Kashi from Ispahan.
<img src="images/pasargad_kashi.png" style="width:750px;height:300px;">

- A scientific study of a turbulent fluid with the style of a abstract blue fluid painting.
<img src="images/circle_abstract.png" style="width:750px;height:300px;">

## Generated Output (Content Image + Style Image = Generated Image)
<img src="z/11.jpg" style="width:500;height:500px;">
<img src="z/22.jpg" style="width:500;height:500px;">
<img src="z/33.jpg" style="width:500;height:500px;">
<img src="z/44.jpg" style="width:500;height:500px;">

## Conclusion
- Neural Style Transfer is an algorithm that given a content image C and a style image S can generate an artistic image
- It uses representations (hidden layer activations) based on a pretrained ConvNet. 
- The content cost function is computed using one hidden layer's activations.
- The style cost function for one layer is computed using the Gram matrix of that layer's activations. The overall style cost function is obtained using several hidden layers.
- Optimizing the total cost function results in synthesizing new images. 


## References:

The Neural Style Transfer algorithm was due to Gatys et al. (2015). Harish Narayanan and Github user "log0" also have highly readable write-ups from which we drew inspiration. The pre-trained network used in this implementation is a VGG network, which is due to Simonyan and Zisserman (2015). Pre-trained weights were from the work of the MathConvNet team. 

- Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, (2015). A Neural Algorithm of Artistic Style (https://arxiv.org/abs/1508.06576) 
- Harish Narayanan, Convolutional neural networks for artistic style transfer. https://harishnarayanan.org/writing/artistic-style-transfer/
- Log0, TensorFlow Implementation of "A Neural Algorithm of Artistic Style". http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style
- Karen Simonyan and Andrew Zisserman (2015). Very deep convolutional networks for large-scale image recognition (https://arxiv.org/pdf/1409.1556.pdf)
- MatConvNet. http://www.vlfeat.org/matconvnet/pretrained/
