# Artificial-Intelligence-Lexical
![](img/README-e591e75a.png)

This repository is aiming to regroup most of the definition in the field of AI. Since there is a lot of things to remember, I would advice the use of flashcard if you are new to the field.

## General terms
### Narrow/Weak Artificial Intelligence

Weak artificial intelligence (weak AI), also known as narrow AI, is artificial intelligence that is focused on one narrow task. Weak AI is defined in contrast to strong AI, a machine with the ability to apply intelligence to any problem, rather than just one specific problem, sometimes considered to require consciousness, sentience and mind). Many currently existing systems that claim to use "artificial intelligence" are likely operating as a weak AI focused on a narrowly defined specific problem.

Siri is a good example of narrow intelligence. Siri operates within a limited pre-defined range of functions. There is no genuine intelligence or no self-awareness despite being a sophisticated example of weak AI.

[Wikipedia](https://en.wikipedia.org/wiki/Weak_AI)

### Artificial General Intelligence (Strong AI/True AI)

Artificial general intelligence (AGI) is the intelligence of a machine that has the capacity to understand or learn any intellectual task that a human being can. It is a primary goal of some artificial intelligence research and a common topic in science fiction and future studies. Some researchers refer to Artificial general intelligence as "strong AI", "full AI" or as the ability of a machine to perform "general intelligent action"; others reserve "strong AI" for machines capable of experiencing consciousness.

Some references emphasize a distinction between strong AI and "applied AI" (also called "narrow AI" or "weak AI"): the use of software to study or accomplish specific problem solving or reasoning tasks. Weak AI, in contrast to strong AI, does not attempt to perform the full range of human cognitive abilities.
[Wikipedia](https://en.wikipedia.org/wiki/Artificial_general_intelligence)

On a side note, here is the 3 theory of consciousness that are taken seriously by the neurology community:
- [global workspace theory](https://en.wikipedia.org/wiki/Global_workspace_theory)
- biological theory of consciousness
- [Higher order theory of consciousness](https://en.wikipedia.org/wiki/Higher-order_theories_of_consciousness)

## Ensemble learning method 
### Random Forest



# ANN (Artificial Neural Network)
![](assets/README-cbf40efe.png)
_________
## Layers
legend
________
## Activation function
The activation function of a node defines the output of that node given an input or set of inputs.
[wikipedia](https://en.wikipedia.org/wiki/Activation_function)

### Sigmoid
![](assets/README-68802d48.png)
![](assets/README-a9b26082.png)
### Tanh
![](assets/README-cd750816.png)
### reLu
### LeakyReLU


### Cross Validation

### Regularisation

### Dropout
**i.e** Dropout is a method that allow you to "turn off" some of the neurones randomly. the most common dropout probability for a layer is 0.5 but it can be tune at will.

![](img/README-3220d9c6.png)

### Overfitting (high variance) and Underfitting (High bias)
![](img/README-9ad36b45.png)

## Gradient descent

![](img/README-be9647f2.png)

### Stochastic gradient descent (SGD)
In both gradient descent (GD) and stochastic gradient descent (SGD), you update a set of parameters in an iterative manner to minimize an error function.

While in GD, you have to run through ALL the samples in your training set to do a single update for a parameter in a particular iteration, in SGD, on the other hand, you use ONLY ONE or SUBSET of training sample from your training set to do the update for a parameter in a particular iteration. If you use SUBSET, it is called Minibatch Stochastic gradient Descent.

![](img/README-b4dc4b9c.png)

Thus, if the number of training samples are large, in fact very large, then using gradient descent may take too long because in every iteration when you are updating the values of the parameters, you are running through the complete training set. On the other hand, using SGD will be faster because you use only one training sample and it starts improving itself right away from the first sample.

SGD often converges much faster compared to GD but the error function is not as well minimized as in the case of GD. Often in most cases, the close approximation that you get in SGD for the parameter values are enough because they reach the optimal values and keep oscillating there.

If you need an example of this with a practical case, check Andrew NG's notes here where he clearly shows you the steps involved in both the cases. cs229-notes
[source](https://datascience.stackexchange.com/questions/36450/what-is-the-difference-between-gradient-descent-and-stochastic-gradient-descent)
______________________________
### Early stopping
## Categories of machine learning

### Unsupervised Learning
Unsupervised learning is a type of self-organized Hebbian learning that helps find previously unknown patterns in data set without pre-existing labels. It is also known as self-organization and allows modelling probability densities of given inputs. It is **one of the main three categories of machine learning, along with supervised and reinforcement learning.**

[Wikipedia](https://en.wikipedia.org/wiki/Unsupervised_learning)
![](img/README-6b811f2b.png)

the four main task of unsupervised learning are:
 - **Dimensionality reduction**

 - **Clustering**

Group similar instance together into *cluster*
 - **Anomaly detection**

Learn what a "normal" data look like, and use it to detect abnormal instances.

 - **Density estimation**

 Estimating how isolated a data is. *i.g* instance located in low density region are likely to be Anomaly. it is usefull for *data analysis, data visualisation and anomaly detection*.

#### Clustering
*(page 237)* Used for :
- Customer segmentation
- Data analysis
- Dimensionality reduction
- Anomaly detection
- Semi-supervised learning
- Search engines (similar image search)
- Segment an image

#### K-Means
- how does it work ?
![](img/README-bb84007a.png)
1) Centroids are assigned randomly

![](img/README-4398e787.png)
- what about it's complexity ?
K-means is **generally linear** with regard to m, k, n. However if the structure does not have a clustering structure it can increase exponentially but it is rarely the case, generally speaking K-Means is one of the fastest clustering algorithms.

- What are the disadvantage?

1) Kmeans tries to create same sized cluster no matter how the data is scattered
2) Kmeans doesnt work well for non-globular structures
3) Kmeans doesnt care about how dense the data is present
4) Curse of dimensionality affects kmeans at high dimension since it uses distance measure

##### Mini batch Kmean
##### elkan Kmean


#### DBSCAN
[Demo](https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/)

Why DBSCAN ?
- Desnsity
- Spatial
- Clustering of
- Application with
- Noise
What input does it take ?
- Set of points
- Neighborhood N
- minpts (density)

How does it work ?

- 1) For each instance, the algorithm count how many instances are located within a small distance "epsilon". this region is called the **epsilon-neighbourhood**.

- 2) If an instance has at least **min_sample** instances in its **epsilon-neighbourhood** (including itself then it is considered a **core instance**) in other word the instance is considered to be in a dense regions.

- 3) All instances in the **neighbourhood** of a **core instance** belong to the same cluster. this neighbourhood may include other core instances; therefore, a long sequence of neighbouring core instances form a single cluster.

- 4) Any instance that is not a core instance and does not have one in it's neighbourhood is **considered an anomaly**.

cons :
- How to choose the density level (i.e min_neighbors) ?
- changing levels means starting from scratch.

##### how to find epsilon ?
In layman’s terms, we find a suitable value for epsilon by calculating the distance to the nearest n points for each point, sorting and plotting the results. Then we look to see where the change is most pronounced (think of the angle between your arm and forearm) and select that as epsilon.
[source](https://towardsdatascience.com/machine-learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python-example-3100091cfbc)

#### Level Set Tree

#### Other Clustering Algorithms
##### Agglomerative clustering
##### BIRCH
##### Mean-Shift
##### Affinity propagation
##### Spectral clustering
#### Dimensionality reduction
Reduce the dimension of a dataset visualisation, to have a better understanding of the data.
##### PCA
PCA select the axis that has the most variation to avoid losing as much information as possible.

the variance ratio is regarded as the amount of information kept during the dimensionality reduction, you generally don't want the variance to go lower than 95% unless it is for data visualisation. In that case use 2 or 3 dimension to have a clear visual of your data
#### Auto encoder
**"official" explenation :**
"An autoencoder is a type of artificial neural network used to learn efficient data coding in an unsupervised manner. The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for dimensionality reduction, by training the network to ignore signal “noise”. Along with the reduction side, a reconstructing side is learnt, where the autoencoder tries to generate from the reduced encoding a representation as close as possible to its original input, hence its name."
[source][https://en.wikipedia.org/wiki/Autoencoder]

**"shortcut" explaination :**
Auto encoder are just neural network where the target output is the input. and with a bottleneck in the middle.

Basicaly it is an "auto compression algorithms" but it goes further than that:
![](assets/README-b941d5fa.png)
#### Denoising Autoencoder
https://towardsdatascience.com/denoising-autoencoders-explained-dbb82467fc2
#### Sparse Autoencoder
https://medium.com/@syoya/what-happens-in-sparse-autencoder-b9a5a69da5c6
https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf
#### Contractive Autoencoder
https://medium.com/@debanjandatta/deep-contractive-auto-encoder-in-keras-aa074d839eb6
https://deepai.org/machine-learning-glossary-and-terms/contractive-autoencoder
#### variational autoencoder
https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73

### Semi-Supervised Learning
Semi-supervised is a hybridization of supervised and unsupervised techniques.

### Supervised Learning
Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs. It infers a function from labelled training data consisting of a set of training examples. In supervised learning, each example is a pair consisting of an input object (typically a vector) and a desired output value (also called the supervisory signal). A supervised learning algorithm analyses the training data and produces an inferred function, which can be used for mapping new examples. An optimal scenario will allow for the algorithm to correctly determine the class labels for unseen instances. This requires the learning algorithm to generalize from the training data to unseen situations in a "reasonable" way (see inductive bias).

The parallel task in human and animal psychology is often referred to as concept learning.
[Wikipedia](https://en.wikipedia.org/wiki/Supervised_learning)
______
## Architectures
### ANN (Artificial Neural Network)
Artificial neural networks (ANN) or connectionist systems are computing systems that are inspired by, but not necessarily identical to, the biological neural networks that constitute animal brains. Such systems "learn" to perform tasks by considering examples, generally without being programmed with any task-specific rules. For example, in image recognition, they might learn to identify images that contain cats by analysing example images that have been manually labelled as "cat" or "no cat" and using the results to identify cats in other images. They do this without any prior knowledge about cats, for example, that they have fur, tails, whiskers and cat-like faces. Instead, they automatically generate identifying characteristics from the learning material that they process.

An ANN is based on a collection of connected units or nodes called artificial neurons, which loosely model the neurons in a biological brain. Each connection, like the synapses in a biological brain, can transmit a signal from one artificial neuron to another. An artificial neuron that receives a signal can process it and then signal additional artificial neurons connected to it.

![](img/README-e1e57001.png)
### Perceptron
![](assets/README-88266d80.png)

![](img/README-eaa240d8.png)

The perceptron is one of the very first algorithms of machine learning, and the most simple artificial neural network.

### MLP (Multilayer Perceptron)
![](assets/README-8899a3d1.png)
[Good video explanation](https://www.youtube.com/watch?v=u5GAVdLQyIg)
### Cross entropy
### Feed Forward (FF)
![](img/README-ba45725d.png)
### Radial Basis Function Network
![](img/README-1c237d52.png)
### CNN (Convolutional Neural Network)
![](img/README-edda8aca.png)
#### Filter / kernel
In a CNN the filters contain the weight
#### Stride
#### Padding
What?
- Padding is a technique that add border of x number of pixel on an image
Why?
- To allowed to do convolution without loosing pixel on the image itself.
how ?
![](img/README-e72318a0.png)
#### Max pooling

**What does it do?**
Applying max pooling on a matrix will reduce the size of the given image/matrix.

**Why ?**
This is done to in part to help over-fitting by providing an abstracted form of the representation. As well, it reduces the computational cost by reducing the number of parameters to learn and provides basic translation invariance to the internal representation.

- it is quicker than convoluting

**How exactly?**
Forward propagation :
Just store the maximum value highlighted by your filter at each step :

![](img/README-0b91a344.png)

Backpropagation:
Resituate the maximum value that were stored precedently in their respective place and zeroed out the other cell.

![](img/README-a2212cf3.png)

Now the question is "ok, but how do I remember their place of I only have the max number?"

In practice, this is achieved by creating a mask that remembers the position of the values used in the first phase, which we can later utilize to transfer the gradients.

**example ?**

![](img/README-67613e2e.png)
![](img/README-d211c064.png)

**Some anecdote?**
There are multiple pooling technique
- Max pooling: The maximum pixel value of the batch is selected.
- Min pooling: The minimum pixel value of the batch is selected.
- Average pooling: The average value of all the pixels in the batch is selected.

![](img/README-1d36af86.png)

max pooling can apparently be replaced with a convolutional layer with increased stride without loss in accuracy. it is described in this paper : https://arxiv.org/pdf/1412.6806.pdf

#### Average pooling

### RNN (Recurrent Neural Network)

**What is a recurrent neural network ?**
- A recurrent neural network use the weight of the previous occurrence.

**How does it work exactly ?**
- During the forward propagation the weight of the hidden layer are the one from the previous layer.

Note: Stochastic gradient descent apparently don't work well with RNN

![](img/README-7ce816c4.png)

#### LSTM (Long Short Term Memory)
![](img/README-f3c75080.png)
[Wikipedia](https://en.wikipedia.org/wiki/Long_short-term_memory) |
[The paper](https://www.bioinf.jku.at/publications/older/2604.pdf)

#### GRU (Gated Recurrent Unit)
![](img/README-8de68771.png)

#### Markov chain
![](img/README-bc1def60.png)
#### Hopfield network

- Category : unsupervised machine learning

The Hopfield network is an [Auto-associative Memory Network](https://en.wikipedia.org/wiki/Autoassociative_memory)

![](img/README-3da1bde9.png)

### Boltzmann Machine Network
- Category : unsupervised machine learning

![](img/README-3c5fa0c8.png)

![](img/README-9411e985.png)
### GAN (Generative adversarial network)
![](img/README-3606fe15.png)

[paper](https://arxiv.org/pdf/1406.2661.pdf)


### DRN
![](img/README-1c865349.png)



# Conclusion

![](img/README-0cc77cbc.png)
https://towardsdatascience.com/the-mostly-complete-chart-of-neural-networks-explained-3fb6f2367464
# other
autoML

## Training Models
### Generalised linear models
#### Linear regression (multiple and single)
![](img/README-cb719ed8.png)

**Linear regression** allow us to put a **straight line** that match the trend of a graph with 2 output (or more with multiple linear regression). It is used to then make prediction.

![](img/README-e80abbd7.png)

What about the training ?
We use Root Mean Square Error or Mean square error.
![](img/README-279baee6.png)

[source](https://machinelearningmastery.com/implement-simple-linear-regression-scratch-python/
)

#### Logistic Regression
Logistic regression is a binary classifier. It is different from linear regression, in the sense that it is used to predict a boolean.

**Logistic  Regression predict whether something is true or false.**
![](img/README-1fe4ebd5.png)

### Polynomial Regression
![polynomial regression compared](img/README-ed7a7eba.png)

# Misc
## Decision Tree
| pro | cons  |
| :------------- | :------------- |
| Require very little data preparation. (no need for scalling or normalization)       | A small change in the data can cause a large change in the structure of the decision tree causing instability.      |
| Missing value does not affect noticeably the result| Often require higher time to train the model |
| Very straightforward result | Cannot handle complicated relationship between feature|

[implementation example](decision_tree/)

![petal decision tree](assets/README-aa81072c.png)
Now at this point it kind of feel like magic. but take a look a the following graph on the picture to have a better intuition of what the algorithm do.

![graph equation decision tree](assets/README-76ac06ce.png)
to clarify a the equation itself here is a [video](https://www.youtube.com/watch?v=atw7hUrg3_8) that worked for me (brave yourself for the Indian accent).

also here is a really good [article](https://towardsdatascience.com/entropy-how-decision-trees-make-decisions-2946b9c18c8) that explain the equation.

# Transformer
The transformer is composed of multiple parts and subparts.

It firstly consists of the encoder, decoder and a final linear layer.

Then in the encoder the first concept is the attention mechanism or "Scaled Dot-Product Attention" as stated in the [paper](https://arxiv.org/pdf/1706.03762.pdf).

The second concept is the "Multi-Head Attention" which is simply the parallelized version of the former

# Self attention

![](README.assets/README-6f921b87.png)

In a nutshell self attention is basically getting the vectors after the word embedding layer and mixing/influencing each vector to one another.

The output is then the same amount of vector/word but each one is now semantically related to the context of the sentence.

![](README.assets/README-60e53346.png)

[video](https://www.youtube.com/watch?v=yGTUuEx3GkA&amp%3Bindex=9)

- What is the difference between **attention** and **self attention** mechanism ?

  [stack](https://datascience.stackexchange.com/questions/49468/whats-the-difference-between-attention-vs-self-attention-what-problems-does-ea)

Visual of what the mechanism is doing.

![](README.assets/README-4e7eba79.png)

![attention animated visual](assets/attention.gif)

# Diet Architecture

![](README.assets/README-ae1cf94c.png)
[video](https://www.youtube.com/watch?v=vWStcJDuOUk&list=PL75e0qA87dlG-za8eLI6t0_Pbxafk-cxb&index=2)
# Question

* Whet is the difference between Artificial Neural Network and deep learning?
  <details>
  <summary>Answer</summary>

  </details>

* What are the pros and cons of supervised vs unsupervised learning ?
  <details>
  <summary>Answer</summary>

  **Supervised**
  - More accurate
  - Labelled data required
  - Requires human in the loop

  **Unsupervised**
  - Less accurate
  - No labelled data required
  - Minimal human effort
  </details>
# Interesting mention
generative design
Geodesign
# Most interesting stack overflow thread I saw:
- [Is running more epoch really a direct cause of Overfitting ?](https://ai.stackexchange.com/questions/17287/is-running-more-epochs-really-a-direct-cause-of-overfitting)
This thread made me realise that if you want to know if you tuned your hyperparameter properly you should run as much epoch as possible. if it does not overfit then you won.

# Most interesting blog posts I saw:
[sanity check before training](https://cs231n.github.io/neural-networks-3/#sanitycheck)
