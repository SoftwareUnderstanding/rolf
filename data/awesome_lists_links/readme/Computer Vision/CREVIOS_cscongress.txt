# cscongress

#### Compiler : 
Colaboratory, or “Colab” for short, is a product from Google Research. Colab allows anybody to write and execute arbitrary python code through the browser, and is especially well suited to machine learning, data analysis and education. More technically, Colab is a hosted Jupyter notebook service that requires no setup to use, while providing free access to computing resources including GPUs.

Colab is able to provide free resources in part by having dynamic usage limits that sometimes fluctuate, and by not providing guaranteed or unlimited resources. This means that overall usage limits as well as idle timeout periods, maximum VM lifetime, GPU types available, and other factors vary over time. Colab does not publish these limits, in part because they can (and sometimes do) vary quickly.

The types of GPUs that are available in Colab vary over time. This is necessary for Colab to be able to provide access to these resources for free. The GPUs available in Colab often include Nvidia K80s, T4s, P4s and P100s. 

#### Language :
- Python 3.7
##### Packages:
- Numpy
- Scipy
- Matlab
- Scikit
- Tensorflow
- Opencv

### Abstract :
Brain tumor is one of the most fearsome medical problems globally today. With the advent of various natural factors, this is among one of those medical problems that is seeing a significant rise in happenstance. Our project aims to utilize the analytical power of neural networking in engineering a method for accurate detection of tumorous tissue in magnetic resonance (MR) images. Based on a type of network known as Deep Neural Network (DNN), our solution proposes, on a limited scale, to identify low and high grade glioblastomas (a type of tumor) pictured in a cerebral MRI scan. These tumors can appear anywhere in the brain, and have variable shapes and contrasts in an MRI, which translates to the requirement of a machine learning solution for detection of these phenomena that exploits a flexible and high capacity DNN, while at the same time, being extremely efficient. Our aim is to produce a rough-edged, efficient, but effective assistive tool that learns as it is used for radiologists’ use in machine-assisted detection of tumors, in order that the physician has a time-effective and accurate tool in their disposal in diagnosis and subsequent treatment of this life-threatening malady.
<br>
For project Vedio : https://www.youtube.com/watch?v=DvP72zne7gQ

#### We used pretrained model Unet : https://arxiv.org/abs/1505.04597

### Dice Score :
Sørensen's original formula was intended to be applied to discrete data. Given two sets, X and Y, it is defined as

<img src="https://latex.codecogs.com/gif.latex?{\displaystyle&space;DSC={\frac&space;{2|X\cap&space;Y|}{|X|&plus;|Y|}}}" title="{\displaystyle DSC={\frac {2|X\cap Y|}{|X|+|Y|}}}" />

where |X| and |Y| are the cardinalities of the two sets (i.e. the number of elements in each set). The Sørensen index equals twice the number of elements common to both sets divided by the sum of the number of elements in each set.

When applied to Boolean data, using the definition of true positive (TP), false positive (FP), and false negative (FN), it can be written as

<img src="https://latex.codecogs.com/gif.latex?{\displaystyle&space;DSC={\frac&space;{2TP}{2TP&plus;FP&plus;FN}}}" title="{\displaystyle DSC={\frac {2TP}{2TP+FP+FN}}}" />

### Activiation :

Activations can either be used through an Activation layer, or through the activation argument supported by all forward layers.
#### relu(rectifier Linear Units) function
Applies the rectified linear unit activation function.
With default values, this returns the standard ReLU activation: max(x, 0), the element-wise maximum of 0 and the input tensor.
The Rectified Linear Unit is the most commonly used activation function in deep learning models. The function returns 0 if it receives any negative input, but for any positive value  x  it returns that value back. So it can be written as  f(x)=max(0,x) .
Graphically it looks like this:
<img src="123.jpg" width="200px">

#### Sigmoid Function
<code>
tf.keras.activations.sigmoid(x)
</code>
<br>Sigmoid activation function, sigmoid(x) = 1 / (1 + exp(-x)).
Applies the sigmoid activation function. For small values (<-5), sigmoid returns a value close to zero, and for large values (>5) the result of the function gets close to 1.
Sigmoid is equivalent to a 2-element Softmax, where the second element is assumed to be zero. The sigmoid function always returns a value between 0 and 1.
<img src='23.png' width="200px">

### Understanding Neral Network 

<img src='tikz14.png' width="500px"> <img src='tikz15.png' width="500px" style="margin-left:70%;margin-top:-5%">
