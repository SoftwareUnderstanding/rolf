# DEEP LEARNING IN TENSORFLOW JS EBOOK

## PART I: INTRODUCTION TO DL IN JS

### C1 - Deep Learning and Javscript

Backpropagation demo scrollytelling illustration: <http://mng.bz/2J4g>
 Stanford CS231 lecture 4 course notes on backpropagation: <http://cs231n>
.github.io/optimization-2/
 Andrej Karpathy’s “Hacker’s Guide to Neural Nets:” <http://karpathy.github>
.io/neuralnets/

## PART II: A GENTLE INTRODUCTION TO TF.JS

### C2 - Simple Linear Regression in TF.JS

(66) Because TensorFlow.js optimizes its computation by scheduling on the GPU, tensors might not always be accessible to the CPU. The calls to dataSync in listing 2.8 tell TensorFlow.js to finish computing the tensor and pull the value from the GPU into the CPU, so it can be printed out or otherwise shared with a non-TensorFlow operation.

To counteract this, we will first normalize our data. This means that we will scale our features so that they have zero mean and unit standard deviation. This type of normal- ization is common and may also be referred to as standard transformation or z-score nor- malization.

(67) const dataMean = data.mean(0); -> Return mean along horizontal axis (mean of each col)

(71) The complete list of callback triggers (in model.fit() function), as of version 0.12.0, is onTrainBegin, onTrainEnd, onEpochBegin, onEpochEnd, onBatchBegin, and onBatchEnd.
Ex:

```js
await model.fit(tensors.trainFeatures, tensors.trainTarget, {
  batchSize: BATCH_SIZE,
  epochs: NUM_EPOCHS,
  callbacks: {
    onEpochEnd: async (epoch, logs) => { await ui.updateStatus(
    `Epoch ${epoch + 1} of ${NUM_EPOCHS} completed.`); trainLoss = logs.loss;
    await ui.plotData(epoch, trainLoss);
    }}
});
```

(72) The idea of val set is: fit model on test set and adjust hyper para based on assessments on val set.

### C3 - Adding Non Linearity Beyond weighted sums

(84) This is because part of the sigmoid function (the part close to the cen- ter) is fairly close to being a straight line
Cascading any number of linear functions always results in a linear function

(87) This brings up a common “gotcha” in building multilayer neural networks: be sure to include nonlinear activations in the hidden layers. Failing to do so results in wasted computation resources and time, with potential increases in numerical instability (observe the wigglier loss curves in panel B of figure 3.4)

(90) Some hyperparameters are:

 The number of dense layers in a model, like the ones in listings 3.1 and 3.2
 What type of initializer to use for the kernel of a dense layer
 Whether to use any weight regularization (see section 8.1) and, if so, the regu-
larization factor
 Whether to include any dropout layers (see section 4.3.2, for example) and, if
so, the dropout rate
 The type of optimizer used for training (such as 'sgd' versus 'adam'; see info
box 3.1)
 How many epochs to train the model for
 The learning rate of the optimizer
 Whether the learning rate of the optimizer should be decreased gradually as
training progresses and, if so, at what rate
 The batch size for training

(91) GridSearch

```js
function hyperparameterGridSearch():
  for units of [10, 20, 50, 100, 200]:
    for learningRate of [1e-5, 1e-4, 1e-3, 1e-2]:
    Create a model using whose dense layer consists of `units` units Train the model with an optimizer with `learningRate`
    Calculate final validation loss as validationLoss
    if validationLoss < minValidationLoss
            minValidationLoss := validationLoss
            bestUnits := units
            bestLearningRate := learningRate
  return [bestUnits, bestLearningRate]
```

(111) However, accuracy is a bad choice for loss function because it suffers from the same zero-gradient issue as the accuracy in binary classification.

```js
function categoricalCrossentropy(oneHotTruth, probs):
  for i in (0 to length of oneHotTruth)
    if oneHotTruth(i) is equal to 1 return -log(probs[i]);
```

### C4 - Recognized Images and Sounds using ConvNet

(121) The groups of conv2d-maxPooling2d layers are the working horse of feature extraction
By passing the input image data through the successive layers of con- volution and pooling, we get tensors that become smaller and smaller in size and more and more abstract in the feature space

(122) Compared to a dense layer, a conv2d layer has more configuration parameters. kernelSize and filters are two key parameters of the conv2d layer

(125) a conv2d layer is “sparsely connected.” While dense layers learn global patterns in the input, convolutional layers learn local patterns—patterns within the small window of the kernel.

(126) A maxPooling2d layer serves two main purposes in a convnet. First, it makes the convnet less sensitive to the exact location of key features in the input image -> positional invariance.

(127) The higher the level, the more abstract the representation and the more removed from the pixel-level values the features are.

(131) the benefit of using larger batch sizes is that it produces a more consistent and less variable gradient update to the model’s weights than a smaller batch size
You should also keep in mind that given the same amount of training data, a larger batch size leads to a small number of gradient updates per epoch. So, if you use a larger batch size, be sure to increase the number of epochs accordingly so you don’t inadvertently decrease the number of weight updates during training

(135) The second way to get image tensors in the browser is to use the TensorFlow.js func- tion tf.browser.fromPixels() on HTML elements that contain image data—this includes img, canvas, and video elements.
Ex: let x = tf.browser.fromPixels( document.getElementById('my-image')).asType('float32');

Resize by using 1 of 2 tf.image.resizeBilinear() or tf.image.resizeNearestNeigbor() (less computationally intensive than bilinear interpolation)
Ex: x = tf.image.resizeBilinear(x, [newHeight, newWidth]);

### C5 - Transfer Learning Reusing the Pretrained Models

#### 1. Transfer Learning the first Approach _(using 2 models)_

(156) We mentioned previously that the compile() call configures the optimizer, loss function, and metrics. However, the method also lets the model refresh the list of weight variables to be updated during those calls.

(157) Note that some of the layers we’ve frozen contain no weights (such as the maxPooling2d layer and the flatten layer) and therefore don’t contribute to the count of nontrain- able parameters when they are frozen.

(158) These differences reflect the advantage of transfer learning: by reusing weights in the early layers (the feature- extracting layers) of the model, the model gets a nice head start relative to learning everything from scratch. This is because the data encountered in the transfer-learning task is similar to the data used to train the original model.

(166) Instead of holding concrete values, a symbolic tensor specifies only a shape and a dtype. A symbolic tensor can be thought of as a “slot” or a “place- holder,” into which an actual tensor value may be inserted later, given that the tensor value has a compatible shape and dtype. (167) It is a “blueprint” for the shape and dtype of the actual tensor values that the model or layer object will output.

(167) In the new approach, we use the tf.model() function, which takes a configuration object with two mandatory fields: inputs and outputs. The inputs field is required to be a symbolic tensor (or, alternatively, an array of symbolic tensors), and likewise for the outputs field. Therefore, we can obtain the symbolic tensors from the original MobileNet model and feed them to a tf.model() call. The result is a new model that consists of a part of the original MobileNet.

(167) The last few layers of a deep convnet are sometimes referred to as the head. What we are doing with the tf.model() call can be referred to as truncating the model. The truncated MobileNet preserves the feature- extracting layers while discarding the head.

(169) This kind of lower-dimension representation of inputs is often referred to as an embedding.

> **_NOTE:_** This will predict the output at layer (n) -> this will become the input of our model when using pretrained. In the layer declaration, remove first dim as this is the batch of training exams.
>
> ```js
>ui.setExampleHandler((label) => {
>  tf.tidy(() => {
>    const img = webcam.capture();
>    controllerDataset.addExample(truncatedMobileNet.predict(img), label);
>    ui.drawThumb(img, label);
>  });
>});
>```

(170) As a result of the two-model setup, it is not possible to train the new head directly using the image tensors (of the shape [numExamples, 224, 224, 3]). Instead, the new head must be trained on the embeddings of the images— the output of the truncated MobileNet. Luckily, we have already collected those embedding tensors

![Transfer Learn Model](readme-imgs/transfer-learn.png)

---

**_The next process code:_**

```js
while (isPredicting) {
  const predictedClass = tf.tidy(() => { const img = webcam.capture();
  const embedding = truncatedMobileNet.predict( img);
  const predictions = model.predict(activation);
  return predictions.as1D().argMax();
});
```

---

(172) One interesting aspect of the method we used in this example is that the training and inference process involves two separate model objects. (177) Another advantage of this approach is that it exposes the embeddings and makes it easier to apply machine-learning techniques that make direct use of these embeddings.

(173) In some cases, embeddings give us vector representations for things that are not even origi- nally represented as numbers (such as the word embeddings in chapter 9).
In cases where the number of reference examples is not too large, and the dimensionality of the input is not too high, using kNN can be computationally more efficient than train- ing a neural network and running it for inference. However, kNN inference doesn’t scale well with the amount of data.

Other ref links: <http://mng.bz/2Jp8>, <http://mng.bz/1wm1>

#### 2. Fine-tuning transfer model _(2nd Approach)_

(177) Fine-tuning is a technique that helps you reach levels of accuracy not achievable just by training the new head of the transfer model.

(178) What does the apply() method do? As its name suggests, it “applies” the new head model on an input and gives you an output.

```js
this.transferHead = tf.layers.dense({
  units: this.words.length,
  activation: 'softmax',
  inputShape: truncatedBaseOutput.shape.slice(1)
});
const transferOutput =
this.transferHead.apply(truncatedBaseOutput) as tf.SymbolicTensor;
this.model =
tf.model({inputs: this.baseModel.inputs, outputs: transferOutput});
```

>**Visual way of the above method:**![Graph for showing apply() to form a new model](./readme-imgs/transfer-learn-new-model.png)

Every inference takes only one predict() call and is therefore a more streamlined process. () This enables us to perform the fine- tuning trick. This is what we will explore in the next section.

(180) Some layers from the transfer model will be unfreezed and contiunuing to receive the updates of the head layer (in fine tuning process)
>![Graph for showing apply() to form a new model](./readme-imgs/fine-tuning.png)
>**_NOTE_**:
*Each time you freeze or unfreeze any layers by changing their trainable attri- bute, you need to call the compile() method of the model again in order for the change to take effect.
*Obviously, if the validation set lacks certain words, it won’t be a very good set to measure the model’s accuracy on. This is why we use a custom function (balancedTrainValSplit in listing 5.8)

(183) So why does fine-tuning help? It can be understood as an increase in the model capacity.

(184) One question you might want to ask is, here we unfreeze only one layer in the base model, but will unfreezing more layers help? The short answer is, it depends, because unfreezing even more layers gives the model even higher capacity. But as we men- tioned in chapter 4 and will discuss in greater detail in chapter 8, higher capacity leads to a higher risk of overfitting ...

#### 3. Object detection first task in book

(186) The nice things about using **synthesized data** are 1) the true label values are automati- cally known, and 2) we can generate as much data as we want. Every time we generate a scene image, the type of the object and its bounding box are automatically available to us from the generation process. (commonly used techniques in DL).

(189) We can build custom loss function for customize loss of certain signature.
>![Graph for building custom obj detection model](./readme-imgs/custom-loss-function.png)

(191) Why do we do this instead of using binary cross entropy as we did for the phishing- detection example in chapter 3? We need to combine two metrics of accuracy here: one for the shape prediction and one for the bounding-box prediction. The latter task involves predicting continuous values and can be viewed as a regression task. Therefore, MSE is a natural metric for bounding boxes. In order to combine the met- rics, we just “pretend” that the shape prediction is also a regression task. This trick allows us to use a single metric function (the tf.metric.meanSquaredError() call in listing 5.10) to encapsulate the loss for both predictions.

(191) By scaling the 0–1 shape indicator, we make sure the shape prediction and bounding-box prediction contribute about equally to the final loss value.

>_**NOTE**_: Build custom loss for evaluate of two task of recognition (classif and return bounding box) at once.
**(194)** Instead of using a single meanSquaredError metric as the loss function, the loss function of a real object-detection model is a weighted sum of two types of losses: 1) a softmax cross-entropy-like loss for the probability scores pre- dicted for object classes and 2) a meanSquaredError or meanAbsolute- Error-like loss for bounding boxes. The relative weight between the two types of loss values is carefully tuned to ensure balanced contributions from both sources of error.
>_**REF link**_:
**<1>** Wei Liu et al., “SSD: Single Shot MultiBox Detector,” Lecture Notes in Computer Science
9905, 2016, <http://mng.bz/G4qD>.
**<2>** Joseph Redmon et al., “You Only Look Once: Unified, Real-Time Object Detection,” Pro- ceedings IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 779–788, <http://mng.bz/zlp1>.
**<3>** Karen Simonyan and Andrew Zisserman, “Very Deep Convolutional Networks for Large-Scale Image Recognition,” submitted 4 Sept. 2014, <https://arxiv.org/abs/1409.1556>.
**<4>** “Understanding SSD MultiBox—Real-Time Object Detection In Deep Learn- ing” by Eddie Forson: <http://mng.bz/07dJ>.
**<5>** “Real-time Object Detection with YOLO, YOLOv2, and now YOLOv3” by Jona- than Hui: <http://mng.bz/KEqX>.

## PART III: ADVANCED DL IN TENSORFLOW.JS

### C6 - Working with data

#### CREATING A TF.DATA.DATASET FROM AN ARRAY

#### CREATING A TF.DATA.DATASET FROM A CSV FILE

(206) In Node.js, we can connect to a local CSV file by using a URL handle with the file:// prefix, like the following:

>```javascript
> const data = tf.data.csv(
>'file://./relative/fs/path/to/boston-housing-train.csv');
>```

#### 1. Using tf.data to manage data

Done by using _tf.data.generator()_.
(207) Generator datasets are powerful and tremendously flexible and allow developers to connect models to all sorts of data-providing APIs, such as data from a database query, from data downloaded piecemeal over the network, or from a piece of con- nected hardware.

(209)

* The first way to access data from a dataset is to stream it all out into an array using _Dataset.toArray()_. The user should use caution when executing this function to not inadver- tently produce an array that is too large for the JavaScript runtime.
* The second way to access data from a dataset is to execute a function on each example of the dataset using dataset.forEachAsync(f).

(210) tf.data.Dataset provides a chainable API of methods to perform these sorts of operations, described in table 6.3. Each of these methods returns a new Dataset object, but don’t be misled into thinking that all the elements of the dataset are cop- ied or that all the elements are iterated over for each method call! The tf.data .Dataset API only loads and transforms elements in a lazy fashion.

>![Table of chainable methods in TF Js](./readme-imgs/table-chainable-methods.png)
>![Table of chainable methods in TF Js](./readme-imgs/table-chainable-methods-2.png)

(212) It is very important that the data is shuffled the same way when we are taking the samples, so we don’t end up with the same example in both sets; thus we use the same random seed for both when sampling both pipelines.

#### 2. Training models with model.fitDataset

(214) Foremost, we don’t need to write code to manage and orchestrate the downloading of pieces of our dataset—this is handled for us in an efficient, as-needed streaming manner.

#### 3. Common patterns for accessing data

(223) Since the Dataset object does not actually contact the remote resource until the data is needed, it’s important to take care to write the error handling in the right place.

(227) The forEach() and toArray() methods should not be used on a webcam iterator

(228) One final note: when using the webcam, it is often a good idea to draw, process, and discard an image before making predictions on the feed. () Similar to the webcam API, the microphone API creates a lazy iterator allowing the caller to request frames as needed, packaged neatly as tensors suitable for consumption directly into a model.

#### 4. Your data is likely flawed: Dealing with problems in your data

(231) If, for instance, our training data and inference data are samples from different distributions, we say there is dataset skew.

(232) Our ideal states that the samples are independent and identically distributed (IID).

(233) Certainly if we know the size of the datasets (17,000 in this case), we can specify the window to be larger than the entire dataset, and we are all set.

(234) The right approach here is to shuffle the entire dataset by using a windowSize >= the number of samples in the dataset.

(235) Outliers are samples in our dataset that are very unusual and somehow do not belong to the underlying distribution.

>![Type of missind data info box](./readme-imgs/type-of-missing-data.png)
>![Type of missind data info box](./readme-imgs/type-of-missing-data-2.png)

(237) Only if your missing data is MCAR are you completely safe to discard samples. () More sophisticated techniques involve building predictors for the missing features from the available features and using those.

(239) A simple way to quickly look at the statistics of your dataset is to use a tool like Facets (<https://pair-code.github.io/facets/>).

(240) In gen- eral, it’s best to z-normalize (normalize the mean and standard deviation of) your data before training.

(241) A good place to get started is the page on responsible AI practices at <https://ai.google/education/responsible-ai-practices>

#### Data Augmentation

(242) As such, this may not be enough to completely get rid of overfitting. Another risk of using data augmentation is that the training data is now less likely to match the distribution of the inference data, introduc- ing skew.

**(243) It’s also important to see that augmentation should not be applied to the validation or testing set.**

### C7 - Visualize data and models

#### 1. Data visualization

##### 1.1. Line Chart

(249) In general, it is good practice to limit the size of the data to be rendered in interactive visualizations for the sake of a smooth and responsive UI.

##### 1.2. Scatter Plot

##### 1.3. Bar chart

(251) The first argument passed to barchart() is not an object consisting of a value field.

##### 1.4. Histograms

(252) A histogram assigns the values into bins. Each bin is simply a continuous range for the value, with a lower bound and an upper bound.

##### 1.5. HeatMap

#### 2. Visualize models after training

3 Most Basic and Useful Techniques

 Visualizing the outputs of intermediate layers (intermediate activations) of a convnet

 Visualizing convnet filters by finding input images that maximally activate them

 Visualizing heatmaps of class activation in an input image

(266) This process is schematically gradient ascent in input space, as opposed to the gradient descent in weight space that underlies typical model training.

![Gradient ascent process](./readme-imgs/gradient-ascent.png)

(268) This is the function defined by the line

```js
const lossFunction = (input) =>
  auxModel.apply(input, {training: true}).gather([filterIndex], 3);
```

Here, auxModel is an auxiliary model object created with the familiar tf.model() function. It has the same input as the original model but outputs the activation of a given convolutional layer.

(269) The question that CAM (Class activation map) aims to answer is “which parts of the input image play the most important roles in causing the convnet to output its top classification decision?

>_**(271) Further Reading and Ref:**_
>
> Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin, “Why Should I Trust You? Explaining the Predictions of Any Classifier,” 2016, <https://arxiv.org/pdf/1602.04938.pdf>.
 TensorSpace (tensorspace.org) uses animated 3D graphics to visualize the topology and internal activations of convnets in the browser. It is built on top of TensorFlow.js, three.js, and tween.js.
 The TensorFlow.js tSNE library (github.com/tensorflow/tfjs-tsne) is an effi- cient implementation of the t-distributed Stochastic Neighbor Embedding (tSNE) algorithm based on WebGL. It can help you visualize high-dimensional datasets by projecting them to a 2D space while preserving the important struc- tures in the data.

### C8 - Underfitting, Overfitting and the universal ML workflow

#### 1. Underfit, overfit and countermeasure

(276) generator/iterator specification of JavaScript (<http://mng.bz/RPWK>)
() As we described in chapter 6, a tf.data.Dataset object, when used in conjunction with the fitDataset() method of a tf.Model object, enables us to train the model even if the data is too large to fit into WebGL memory (or any applicable backing memory type) as a whole.

(280) To over- come underfitting, we usually increase the power of the model by making it bigger.
() In particular, the kernelRegularizer and dropoutRate parameters are the ways in which we’ll combat overfitting later.

(282) With a regularized weight, the loss of the model includes an extra term. In pseudo-code,
loss = meanAbsoluteError(targets, prediciton) + 12Rate * 12(kernel)

Here, l2Rate * l2(kernel) is the extra L2-regularization term of the loss function. Unlike the MAE, this term does not depend on the model’s predictions.

As a result, the training process will try to not only minimize the target-prediction mismatch but also reduce the sum of the squares of the kernel’s elements.

(283) Since the L2 regularizer works by encouraging the kernel of the hidden dense layer to have smaller values,

>**Visualizing the weight distribution of layers (from jena-weather/index.js)**

```js
>function visualizeModelLayers(tab, layers, layerNames) { layers.forEach((layer, i) => {
    const surface = tfvis.visor().surface({name: layerNames[i], tab});
    tfvis.show.layer(surface, layer);
    });
}
```

>An intuitive way to understand this is that L2 regularization enforces the principle of **Occam’s razor**. Generally speaking, a larger magnitude in a weight param- eter tends to cause the model to fit to fine-grained details in the training features that it sees, and a smaller magnitude tends to let the model ignore such details.

#### 2. The Universal Workflow of ML

(288) Keep in mind that machine learning can only be used to learn patterns that are present in the training data. In this case, getting up-to-date data and continuously training new models will be a via- ble solution.

### C9 - Deep Learning for sequences and text

#### 1. Intro To RNN

(297) Comparing the structure of simpleRNN and that of the dense layer (figure 9.1), we can see two major differences:
 SimpleRNN processes the input elements (time steps) one step at a time. This reflects the sequential nature of the inputs, something a dense layer can’t do.
 In simpleRNN, the processing at every input time step generates an output (yi).
The output from a previous time step (for example, y1) is used by the layer when it processes the next time step (such as x2).

(298) You can see that the output at time step i becomes the “state” for the next time step (next iteration). State is an important concept for RNNs. It is how an RNN “remembers” what happened in the steps of the input sequence it has already seen.
() SimpleRNN is appropriately named because its output and state are the same thing.
() Furthermore, the for loop reflects another important property of RNNs: parameter sharing.
() While conv2d layers exploit the transla- tional invariance along spatial dimensions, RNN layers exploit translational invariance along the time dimension.

**GRU - Gate Recurrent Unit**
(302) GRU and LSTM are RNNs designed to solve the vanishing-gradient problem, and GRU is the simpler of the two.

(304) For instance, suppose we’re trying to classify a movie review as positive or negative. The review may start by saying “this movie is pretty enjoyable,” but halfway through the review, it then reads “how- ever, the movie isn’t as good as other movies based on similar ideas.” At this point, the memory regarding the initial praise should be largely forgotten, because it is the later part of the review that should weigh more in determining the final sentiment-analysis result of this review.

() The important thing to remember is that the internal structure of GRU allows the RNN to learn when to carry over old state and when to update the state with information from the inputs. This learning is embodied by updates to the tunable weights, Wz, Uz, Wr, Wr, W, and U (in addition to the omitted bias terms).

#### 2. Building DL models for texts

##### 2.1. One hot multi hot

>![One Hot Multi Hot Illustration](./readme-imgs/one-hot-mulit-hot.png)

##### 2.2. Words Embeddings

Just like one-hot encoding (figure 9.6), word embedding is a way to represent a word as a vector (a 1D tensor in TensorFlow.js). However, word embeddings allow the values of the vector’s elements to be trained, instead of hard- coded according to a rigid rule such as the word-to-index map in one-hot encoding.

>![Word Embeddings Illustration](./readme-imgs/word-embeddings.png)

Word embedding gives us the following benefits:
 It addresses the size problem with one-hot encodings. embeddingDims is usually much smaller than vocabularySize.
 By not being opinionated about how to order the words in the vocabulary and by allowing the embedding matrix to be trained via backpropagation just like all other neural network weights, word embeddings can learn semantic rela- tions between words. Words with similar meanings should have embedding vec- tors that are closer in the embedding space.
Also, the fact that the embedding space has multiple dimensions (for example, 128) should allow the embedding vectors to capture different aspects of words.

##### 2.4. 1D ConvNet

>![1D ConvNet Illustration](./readme-imgs/1d-convnet.png)

(314) Why do we need to do truncation and padding? TensorFlow.js models require the inputs to fit() to be a tensor, and a tensor must have a concrete shape.

(317) However, it needs to be pointed out that a conv1d layer by itself is not able to learn sequential patterns beyond its kernel size.
The slower performance of LSTM and RNNs is related to their step-by-step internal operations, which cannot be parallelized; convolutions are amenable to par- allelization by design.

(319) These files can be uploaded to the Embedding Projector (<https://projector.tensorflow> .org) for visualization.
The Embedding Projector tool provides two algorithms for dimension reduction: t-distrib- uted stochastic neighbor embedding (t-SNE) and principal component analysis (PCA)
One of the best known pretrained word- embedding sets is GloVe (for Global Vectors) by the Stanford Natural Language Pro- cessing Group (see <https://nlp.stanford.edu/projects/glove/>).

#### 3. Seq2Seq with attention mechanism

Some variety of Seq2seq tasks:
 Text summarization
 Machine translation
 Word prediction for autocompletion
 Music composition
 Chat bots

![Model for attention mechanism](./readme-imgs/attention-mech.png)

(326) The role of the attention mechanism is to enable each output character to “attend” to the correct characters in the input sequence.

![Attention mechanism deep dive](./readme-imgs/attention-deep-dive.png)

(329) The attention is a dot product (element-by-element product) between the encoder LSTM’s output and the decoder LSTM’s output, followed by a softmax activation:

```js
let attention = tf.layers.dot({axes: [2, 2]}).apply([decoder, encoder]);
attention = tf.layers.activation({ activation: 'softmax',
name: 'attention'
}).apply(attention);
```

>**Materials for further reading**
 Chris Olah, “Understanding LSTM Networks,” blog, 27 Aug. 2015, <http://mng.bz/m4Wa>.
 Chris Olah and Shan Carter, “Attention and Augmented Recurrent Neural Net- works,” Distill, 8 Sept. 2016, <https://distill.pub/2016/augmented-rnns/>.
 Andrej Karpathy, “The Unreasonable Effectiveness of Recurrent Neural Net- works,” blog, 21 May 2015, <http://mng.bz/6wK6>.
 Zafarali Ahmed, “How to Visualize Your Recurrent Neural Network with Atten- tion in Keras,” Medium, 29 June 2017, <http://mng.bz/6w2e>.
 In the date-conversion example, we described a decoding technique based on argMax(). This approach is often referred to as the greedy decoding technique because it extracts the output symbol of the highest probability at every step. A popular alternative to the greedy-decoding approach is beam-search decoding, which examines a larger range of possible output sequences in order to deter- mine the best one. You can read more about it from Jason Brownlee, “How to Implement a Beam Search Decoder for Natural Language Processing,” 5 Jan. 2018, <https://machinelearningmastery.com/beam-search-decoder-natural>- language-processing/.
 Stephan Raaijmakers, Deep Learning for Natural Language Processing, Manning Publications, in press, www.manning.com/books/deep-learning-for-natural- language-processing.

### C10 - Generative Deep Learning

### C11 - Basics of deep reinforcement learning

![Basic RL visulization](readme-imgs/basic-rl.png)

#### 2. Policy Network and Policy Gradients

(379) A natural solution is to build a neural network to select an action based on the observation. This is the basic idea behind the policy network.

![Policy network model](readme-imgs/policy-network.png)

() The reason we don’t include the sigmoid nonlinearity directly in the last (output) layer of the policy network is that we need the logits for training, as we’ll see shortly.

(380) Why do we prefer the more complicated random- sampling approach with tf.multinomial() over this simpler approach? The answer is that we want the randomness that comes with tf.multinomial().

(381) In other words, it has to “learn swimming by swimming,” a key feature of RL problems.

(382) We want to assign higher scores to the actions in the early and middle parts of the episode and assign lower ones to the actions near the end.
This brings us to the idea of reward discounting, a simple but important idea in RL that the value of a certain step should equal the immediate reward plus the reward that is expected for the future.

>**_Policy Gradient:_** It compares the logits (unnormal- ized probability scores) and the actual action selected at the step and returns the gradient of the discrepancy between the two with respect to the policy network’s weights.
The gradients, together with the rewards from the training episodes, form the basis of our RL method. This is why this method belongs to the family of RL algorithms called policy gradients.

(384) During training, we let the agent play a number of games (say, N games) and collect all the discounted rewards according to equation 11.1, as well as the gradients from all the steps. Then, we combine the discounted rewards and gradients by multiplying the gradients with a normalized version of the discounted rewards. The reward normaliza- tion here is an important step. It linearly shifts and scales all the discounted rewards from the N games so that they have an overall mean value of 0 and overall standard deviation of 1. An example of applying this normalization on the discounted rewards is shown in figure 11.6. It illustrates the normalized, discounted rewards from a short episode (length = 4) and a longer one (length = 20). From this figure, it should be clear what steps are favored by the REINFORCE algorithm: they are the actions made in the early and middle parts of longer episodes. By contrast, all the steps from the shorter (length-4) episode are assigned negative values. What does a negative normal- ized reward mean? It means that when it is used to update the policy network’s weights later, it will steer the network away from making a similar choice of actions given similar state inputs in the future. This is in contrast to a positive normalized reward, which will steer the policy network toward choosing similar actions given simi- lar inputs in the future.
![Policy Gradients Illustration](readme-imgs/policy-gradients.png)

>Illustrate for the training process of a Policy Net:
![Policy net train ill](readme-imgs/policy-net-train.png)

(388) If the char- acteristics of the system change over time, we won’t have to derive new mathematical solutions from scratch: we can just re-run the RL algorithm and let the agent adapt itself to the new situation.

#### 3. Value network and Q-Learning - the snake game

**_Frame:_** A step in gaming terminology

(390) One key challenge in the snake game is the snake’s growth. With the length-growth rule, however, the agent must learn to avoid bumping into its own body, which gets harder as the snake eats more fruit and grows longer.
This sparse and complex reward structure is also the main reason why the policy gradient and REINFORCE method will not work well on this problem.
_The policy-gradient method works better when the rewards are frequent and simple, as in the cart-pole problem._

>_**MDP requirement:** The state of the environment at the next step is determined completely by the state and the action taken by the agent at the current step. In other words, MDP assumes that your history (how you got to your current state) is irrelevant to deciding what you should do next._

non-MDP: require historical process -> lots of computational res.

(394) Markov Chain Process Illustration of state and action
>![MDP state and action](readme-imgs/markov-chain-process.png)

(394) A Q-value, denoted Q(s, a), is a function of the current state (s) and the action (a).

(395) Therefore, the RL problem of finding the best decision-making process is reduced to learning the function Q(s, a). This is why this learning algorithm is called Q-learning.

* Policy gradient is about predicting the best action;
* Q-learning is about predicting the values of all possible actions (Q-values).

_**Bellman Equation**_
>![Bellman Equation](./readme-imgs/bellman-equation.png)

() The programmer in you will immediately notice that the Bellman equation (equation 11.3) is recursive: all the Q-values on the right-hand side of the equation can be expanded further using the equation itself.
_The beauty and power of Bellman equation is that it allows us to turn the Q-learning problem into a supervised learning problem,
even for large state spaces.

_**DEEP Q-NETWORK**_

![Snake board observation convert](readme-imgs/snake-board-convert.png)
>![DQN in action](readme-imgs/dqn-process-illustrate.png)

Why it make sense to use NN as the function Q(s,a) in this problem?

* If using lookup table approach -> Too many possible board configurations.
* NN doesn't need to see all possible inputs, it learns to interpolate between training examples through generalization.

_**TRAINING DEEP Q-NET**_

(399) We will train our DQN by pressuring it to match the Bellman equation. If all goes well, this means that our DQN will reflect both the immediate rewards and the optimal dis- counted future rewards.

() Computing samples of input requires the current state si and the action we took at that state, aj, both of which are directly available in the game history.

(400) The random-play part is easily achieved using a random-number generator. The remembering part is achieved with a data structure known as replay memory.
How the replay memory work:

---
>1 si, observation of the current state at step i (the board configuration).
2 ai, action actually performed at the current step (selected either by the DQN as
depicted in figure 11.12 or through random selection).
3 ri, the immediate reward received at this step.
4 di, a Boolean flag indicating whether the game ends immediately after the cur-
rent step. From this, you can see the fact that the replay memory is not just for a single episode of the game. Instead, it concatenates the results from multiple game episodes. Once a previous game is over, the training algorithm simply starts a new one and keeps appending the new records to the replay memory.
5 si+1, the observation from the next step if di is false. (If di is true, a null is stored as the placeholder.)
---

The Replay Memory can be thought as a "dataset" for the DQN. But it kept update as the training goes on.
>![Replay Memory Along Timestamp Illustration](./readme-imgs/replay-memory.png)

_**EPSILON-GREEDY ALGORITHM**_

```js
x = Sample a random number uniformly between 0 and 1.
    if x < epsilon:
      Choose an action randomly
else:
qValues = DQN.predict(observation)
Choose the action that corresponds to the maximum element of qValues
```

(402) In RL problems based on the epsilon-greedy policy, the initial and final values of epsilon are tunable hyperparameters, and so is the time course of epsilon’s down-ramping.

Illustration how the predicted Q-values are extracted from the replay memory in a step of the DQN training.
![Extract Q Value from DQN traning](readme-imgs/q-value-extract.png)

Illustration how the target Q-value using the Bellman Equation.
![Extract Q Value from Bellman Equation](./readme-imgs/bellman-q-value-extract.png)

(406) As you may have noticed, an important trick in the deep Q-learning algorithm here is the use of two instances of DQNs. They are called the online DQN and the target DQN, respectively.

(407) Calc gradient desc using tf.variableGrads() function.
![Loss function for Q Value  predict and backprop](./readme-imgs/dqn-mse-backprop.png)

(409) To improve the snake agent further, we need to tweak the epsilon-greedy algorithm to encourage the snake to explore better moves when its length is long. (<https://github.com/carsonprindle/OpenAIExam2018>.)

>**Materials for further reading**
 Richard S. Sutton and Andrew G. Barto, Reinforcement Learning: An Introduction, A Bradford Book, 2018.
 David Silver’s lecture notes on reinforcement learning at University College London: <http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html>.
 Alexander Zai and Brandon Brown, Deep Reinforcement Learning in Action, Man- ning Publications, in press, www.manning.com/books/deep-reinforcement- learning-in-action.
 Maxim Laplan, Deep Reinforcement Learning Hands-On: Apply Modern RL Methods, with Deep Q-networks, Value Iteration, Policy Gradients, TRPO, AlphaGo Zero, and More, Packt Publishing, 2018.

## PART IV: SUMMARY

### C12 - Testing, Optimizing, and Deploying models

#### 1. Testing the Tensorflow JS models

#### 2. Deploying the TF JS models on various platform and env

(439) One common development mistake is to forget to account for Cross-Origin Resource Sharing (CORS) when setting up a bucket in GCP () but after the initial load, the model can be loaded much faster from the browser cache.

(440) One nice property of web deployment is that prediction happens directly within the browser.
A downside of making predictions within the browser is model security. Sending the model to the user makes it easy for the user to keep the model and use it for other purposes.

() 2 approaches to serving models as cloud serve:

* Serve on a NodeJS server with native JS.
* Using TFX <https://www.tensorflow.org/tfx/guide/serving>. Convert model to SavedModel format:

```bash
tensorflowjs_converter \
--input_format=tfjs_layers_model \
--output_format=keras_saved_model \
/path/to/your/js/model.json \
/path/to/your/new/saved-model
```

() Deploy model as chrome extension
The model secu- rity story and data privacy story are identical to the web page deployment.
Note that the model running in the browser extension has access to the same hard- ware acceleration as the model running in the web page and, indeed, uses much of the same code.

>**Materials for further reading**
 Denis Baylor et al., “TFX: A TensorFlow-Based Production-Scale Machine Learning Platform,” KDD 2017, www.kdd.org/kdd2017/papers/view/tfx-a- tensorflow-based-production-scale-machine-learning-platform.
 Raghuraman Krishnamoorthi, “Quantizing Deep Convolutional Networks forEfficient Inference: A Whitepaper,” June 2018, <https://arxiv.org/pdf/> 1806.08342.pdf.
 Rasmus Munk Larsen and Tatiana Shpeisman, “TensorFlow Graph Optimiza- tion,” <https://ai.google/research/pubs/pub48051>.

### C13 - Summary, Conclusion

#### 2. The universal workflow of supervised deep learning

(458) These difficult steps include understanding the problem domain well enough to be able to determine what sort of data is needed, what sort of predictions can be made with reasonable accuracy and generalization power, how the machine-learning model fits into the overall solution that addresses a practical problem, and how to measure the degree to which the model succeeds at doing its job.

>_**1. Determine if machine learning is the right approach.**_ First, consider if machine learn- ing is the right approach to your problem, and proceed with the following steps only if the answer is yes. In some cases, a non-machine-learning approach will work equally well or perhaps even better, at a lower cost.
_**2. Define the machine-learning problem.**_ Determine what sort of data is available and what you are trying to predict using the data.
_**3. Check if your data is sufficient.**_ Determine if the amount of data you already have is sufficient for model training. You may need to collect more data or hire peo- ple to manually label an unlabeled dataset.
_**4. Identify a way to reliably measure the success of a trained model on your goal.**_ For simple tasks, this may be just prediction accuracy, but in many cases, it will require more sophisticated, domain-specific metrics.
_**5. Prepare the evaluation process.**_ Design the validation process that you’ll use to eval- uate your models. In particular, you should split your data into three homoge- neous yet nonoverlapping sets: a training set, a validation set, and a test set. The validation- and test-set labels ought not to leak into the training data. For instance, with temporal prediction, the validation and test data should come from time intervals after the training data. Your data-preprocessing code should be covered by tests to guard against bugs (section 12.1).
_**6. Vectorize the data.**_ Turn your data into tensors, or n-dimensional arrays—the lin- gua franca of machine-learning models in frameworks such as TensorFlow.js and TensorFlow. You often need to preprocess the tensorized data in order to make it more amenable to your models (for example, through normalization).
_**7. Beat the commonsense baseline.**_ Develop a model that beats a non-machine-learning baseline (such as predicting the population average for a regression problem or predicting the last datapoint in a time-series prediction problem), thereby demonstrating that machine learning can truly add value to your solution. This may not always be the case (see step 1).
_**8. Develop a model with sufficient capacity.**_ Gradually refine your model architecture by tuning hyperparameters and adding regularization. Make changes based on the prediction accuracy on the validation set only, not the training set or the test set. Remember that you should get your model to overfit the problem (achieve a better prediction accuracy on the training set than on the validation set), thus identifying a model capacity that’s greater than what you need. Only then should you begin to use regularization and other approaches to reduce overfitting.
_**9. Tune hyperparameters.**_ Beware of validation-set overfitting when tuning hyperpa- rameters. Because hyperparameters are determined based on the performance on the validation set, their values will be overspecialized for the validation set and therefore may not generalize well to other data. It is the purpose of the test set to obtain an unbiased estimate of the model’s accuracy after hyperparame- ter tuning. So, you shouldn’t use the test set when tuning the hyperparameters.
_**10. Validate and evaluate the trained model.**_ As we discussed in section 12.1, test your model with an up-to-date evaluation dataset, and decide if the prediction accu- racy meets a predetermined criterion for serving actual users. In addition, per- form a deeper analysis of the model’s quality on different slices (subsets) of the data, aiming at detecting any unfair behaviors (such as vastly different accura-cies on different slices of the data) or unwanted biases. Proceed to the final step only if the model passes these evaluation criteria.
_**11. Optimize and deploy the model.**_ Perform model optimization in order to shrink its size and boost its inference speed. Then deploy the model into the serving envi- ronment, such as a web page, a mobile app, or an HTTP service endpoint (sec- tion 12.3).

---
>**Review of models and layer types in TF js**_
>
>1. Densely connected networks and multilayer perceptrons: Such networks are specialized for unordered vector data (for example, the numeric features in the phishing-website-detection problem and the housing- price-prediction problem). Each dense layer attempts to model the relation between all possible pairs of input features and the layer’s output activations. This is achieved through a matrix multiplication between the dense layer’s kernel and the input vector. MLPs are most commonly used for categorical data.
>
>2. Convolutional Network: Convolutional layers look at local spatial patterns by applying the same geometric transformation to different spatial locations (patches) in an input tensor.
>
>3. Recurrent Neural Network: RNNs work by processing sequences of inputs one timestamp at a time and maintain- ing a state throughout. A state is typically a vector or a set of vectors (a point in a geo- metric space). RNNs should be used preferentially over 1D convnets in the case of sequences in which the patterns of interest are not temporally invariant (for instance, time-series data in which the recent past is more important than the distant past).
Three RNN layer types are available in TensorFlow.js: simpleRNN, GRU, and LSTM. For most practical purposes, you should use either GRU or LSTM. LSTM is the more powerful of the two, but it is also computationally more expensive. You can think of GRU as a simpler and cheaper alternative to LSTM.
>
>4. Some pretrained model net:
>
>* Face API
>* Hands Free (Hand Track)
>* Pose Net (Detect Skeletal key points)
>* Speech Commands (18 English spoken words)
>* Toxicity (for English toxic words)
>* Universal sentence encoder
>* ml5js
>* magenta/music
>
---

**Possibilities Application of DL:**
(468 - 469)  Mapping vector to vector
– Predictive healthcare—Mapping patient medical records to predicted treat-
ment outcomes
– Behavioral targeting—Mapping a set of website attributes to a potential
viewer’s behavior on the website (including page views, clicks, and other
engagements)
– Product quality control—Mapping a set of attributes related to a manufactured
product to predictions about how well the product will perform on the mar-
ket (sales and profits in different areas of the market)
 Mapping image to vector
– Medical image AI—Mapping medical images (such as X-rays) to diagnostic results
– Automatic vehicle steering—Mapping images from cameras to vehicle control signals, such as wheel-steering actions
– Diet helper—Mapping images of foods and dishes to predicted health effects (for example, calorie counts or allergy warnings)
– Cosmetic product recommendation—Mapping selfie images to recommended cosmetic products
 Mapping time-series data to vector
– Brain-computer interfaces—Mapping electroencephalogram (EEG) signals to user intentions.
– Behavioral targeting—Mapping past history of product purchases (such as movie or book purchases) to probabilities of purchasing other products in the future
– Prediction of earthquakes and aftershocks—Mapping seismic instrument data sequences to the predicted likelihoods of earthquakes and ensuing after- shocks
 Mapping text to vector
– Email sorter—Mapping email content to generic or user-defined labels (for
example, work-related, family-related, and spam)
– Grammar scorer—Mapping student writing samples to writing-quality scores
– Speech-based medical triaging—Mapping a patient’s description of illness to the
medical department that the patient should be referred to
 Mapping text to text
– Reply-message suggestion—Mapping emails to a set of possible response mes- sages
– Domain-specific question answering—Mapping customer questions to auto- mated reply texts
– Summarization—Mapping a long article to a short summary
 Mapping images to text
– Automated alt-text generation—Given an image, generating a short snippet of text that captures the essence of the content
– Mobility aids for the visually impaired—Mapping images of interior or exterior surroundings to spoken guidance and warnings about potential mobility haz- ards (for example, locations of exits and obstacles)
 Mapping images to images
– Image super-resolution—Mapping low-resolution images to higher-resolution
ones
– Image-based 3D reconstruction—Mapping ordinary images to images of the
same object but viewed from a different angle
 Mapping image and time-series data to vector
– Doctor’s multimodal assistant—Mapping a patient’s medical image (such as an MRI) and history of vital signs (blood pressure, heart rate, and so on) to pre- dictions of treatment outcomes
 Mapping image and text to text
– Image-based question answering—Mapping an image and a question related to
it (for instance, an image of a used car and a question about its make and
year) to an answer
 Mapping image and vector to image
– Virtual try-on for clothes and cosmetic products—Mapping a user’s selfie and a vector representation of a cosmetic or garment to an image of the user wear- ing that product.

#### 3. Trends in DL (473)
