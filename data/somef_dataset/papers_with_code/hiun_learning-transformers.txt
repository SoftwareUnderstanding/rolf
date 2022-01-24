# learning-transformers
This repository contains a fork of the Transformer model from https://github.com/huggingface/transformers. Unlike the original source code, which is a library, this repository is runnable stand-alone code for language modeling tasks.

The source code on such open source libraries are great, however, sometimes it is difficult to read all code for simply running and experimenting with a model. For example, preprocessing data, define a train-eval loop, integrating a model into that loop, these tasks are essential to write machine learning programs but it not always easy to looking source code from a large open source project.

This repository combines open source model code and tutorial level preprocessing, and train-eval code. As a result, we expect readers can easily understand how Transformer models are implemented and work from scratch.

This repository partly uses source code from the following sources:
## reference materials

- Preprocessing, Train-Eval loop (except model): https://pytorch.org/tutorials/beginner/transformer_tutorial.html

- Transformer model: https://github.com/huggingface/transformers/blob/a7d46a060930242cd1de7ead8821f6eeebb0cd06/src/transformers/models/bert/modeling_bert.py


## informal description of transformers model
What is a model? What is their role? I think a good model needs to have a good [Inductive Bias](https://en.wikipedia.org/wiki/Inductive_bias), which means it has good generalization capability to unseen example during training.

The difference between the Neural Network method of learning and other learning paradigm is that the Neural Network method learns from data by making a good representation of that data. On the contrary, many other methods learn by features that are manually selected by humans.

The Transformer model is one of the most popular representation generators of Neural Network methods of learning. Because of its general representation processing mechanism such as Attention-based learning, many recent advancements of deep learning rely on it.

So what actually Transformers do? What modules comprise Transformers? What are their implications? This is a natural question of mine as a beginner.

## What Transformers do? (What Models do?)
Because the Transformer model is a special case of the deep learning model. Let's look at what the Deep Learning model generally does. The Deep Learning model can be viewed as [Directed Acyclic Graph](https://en.wikipedia.org/wiki/Directed_acyclic_graph) (DAG; a graph structure that has no connection to ancestry node).

<center>
<img src="./assets/dag.png" width="400" />
</center>

Deep learning model process matrix, which roles a computable representation of input (such as text, image, etc.). The computation of the Deep Learning model to retrieve more abstract representation of input representation.

<center>
<img src="./assets/imagenet.png" width="500" />
</center>

As [example shows in imagenet](https://towardsdatascience.com/transfer-learning-and-image-classification-using-keras-on-kaggle-kernels-c76d3b030649), the earlier part of DAG (a matrix transformation 'layer' of Deep Learning model) do generate low-level features (e.g. shape of stroke that consists cars) from input data. While the latter part of DAG does transform them into high-level features (e.g. wheels of a car or this is Ford Model T).

<center>
<img src="./assets/self-attention.png" width="700" />
</center>

We can [observe the same for natural language processing](https://nlp.seas.harvard.edu/2018/04/03/attention.html). The above example shows a step in making a representation of an input sentence for a Machine Translation task using the Transformer Model. The `Encoder Layer 2` shows attention in the earlier part of DAG. It visualizes what to Attend for given sentences in given sentences. The diagonal line shows most of the words are self-attended since it is natural that low-level features of the text are in word-level. However, in the latter part of DAG, we show the trend that attention is concentrated on specific words, which refers to an abstract representation of given sentences that are crucial for a given task (such as word 'bad' is important for movie review sentiment analysis)


You may wonder about rules for shaping abstract representation, that is the role of the learning algorithm.
For example, Neural Network is a Learning algorithm that depends on its task setup.
A task like classifying text or image can be viewed as a classic Supervised Learning algorithm.
In Supervised Deep Learning cases, shaping abstract representation is the role of optimization methods (e.g. backpropagation with learning rate scheduling) that utilize label data.
More specifically in the training model, when the model outputs inference result, that is compared against the label, and the difference is backpropagated through the model, this enforces (thus trains) model parameters to generate more task-oriented representation. (Note that the task can be multiple, in the case of multi-task learning. Or task can be much more general, in the case of few-shot learning.)
     

## Modules of Transformers and their Implications

Previously, we revisit the single significant property of the machine learning model that good models have good inductive bias performance to achieve good generalization performance.
Deep learning is Neural Network methods of learning, focusing on learning thus able to deriving task-oriented representation from data.
The Transformer is one of the popular methods of Deep learning that is good at learning textual or visual representation from data.
The Deep Learning model can be viewed as matrix processing DAG; where input data is matrix and DAG is gradually learning representation of data from its low to high-level features.
High-level features are determined by a learning algorithm, especially backpropagation with label data. 
 
I think all of the above parts have meaning to build good machine learning algorithms and systems. Let's look at how the above concepts reflect in the Deep Learning model (the DAG) of Transformers Encoder.

### Model Architecture

![arch](./assets/arch.png)
The above figure shows the Transformer model and its mapping between Python classes.
We only look at the Transformer encoder, because the encoder and decoder share core functionality DAGs. 

### Input Embedding Layer
The role of the input embedding layer is to create a meaningful (task-oriented) representation of sentences - embeddings - from natural language. In this example, we receive word ids and create a 768 dimension vector.
The 768 dimension vector - the embeddings is a low-level representation of sentences, it is shaped by a mixture of the literal representation of text w and weight defined by backpropagation for objectives of the task.
The embedding layer concatenates embedding from other data, such as token type and positional information.
To add regularization for embedding, it applies layer normalization and Dropout to reduce training time and improve generalization respectively.
This is implemented in `BertEmbedding` Python Class.

### Model Scaffolding
Multiple layers of transformation are required to make a high-level representation.
The `BertEncoder` class stacks multiple self-attention layers of a transformer to implement this.
It initializes the transformer layer to input matrix one another.

### Inside of Single Model
The single transformer model is comprised of the attention part, intermediate processing part, and the output processing part.
The first module `BertAttention` applies self-attention and by call `BertSelfAttention`and normalizes its output with `BertSelfOutput` module.
The second module `BertIntermediate` layer is for increasing model capacity, by adding one sparser layer (`hidden_size` 768 dim to `intermediate_size` 3072 dim).
The last layer `BertOutput` is for bottleneck layer (`intermediate_size` 3072 dim to `hidden_size` 768 dim again) and do additional regularization.

`BertLayer` connects above three layers, to be stackable.

### Self-Attention

<center>
<img src="./assets/seq2seq.png" width="250" />
</center>

I think the concept of attention no more than just matrix transformation using *representations from relevant information sources*.
Above figure shows [early work on attention](https://arxiv.org/abs/1409.0473).
In this case, the sequential encoder (RNN) generates a matrix in a bidirectional context, and their output probabilities are concatenated and applied to a given step in the decoding process for making relevant representation for a given task.

Informally, attention is just a probability-contained matrix to *scale* representation to be more appropriate for the task.
Attention is a signal for adjustment of representation, thus it reduces the model size to achieve the same quality performance.

In terms of viewing attention by *scale* representation, what is a good information source for such scaling?
The idea of self-attention is finding the importance of each word and score them, and apply them to scale representation for a given task.
The mechanism of finding scores are multiply query and key matrix.
More specifically it obtains attention score matrix by element-wise multiplication result between single query word and all key words.
Finally, the value vector is scaled by the respected attention score of its word.
A higher score implies that given words have a strong relation to words on those sentences, the training procedures reveal a neural network to define a relationship by adjusting weights which are guided by loss function through backpropagation.
Attention score totally depends on representation, which is the result of linear transformations from the embedding. 
In the training setting, weights of such embedding and linear transformations will be adjusted for generating maximum attention score helpful to resolve the given task.


`BertSelfAttention` class implements this. It initializes, a linear layer of query, key, and value.
Perform matrix multiplication with a query and key value.
Scaling it and make probabilities using Softmax function before applying to value.
Value is an adjusted representation for specific tasks.
Note that the adjustment is gradually refined across stacked layers.


![self-attention-vis](./assets/self-attention-vis.png)

[Above image](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb) shows encoder self-attention (input-input) for machine translation. The image illustrates words related to `it_`. At layer 1, the word has a relationship to most words however, as layer getting compute abstract representation, it shows a strong relationship to a noun like `animal_` or `street_`.


Note that the multi-head means just splitting input representation by number of heads,
To allows the model to attend different subspaces of representation without averaging it ([ref](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#multi-head-self-attention) and [original paper](https://arxiv.org/abs/1706.03762)).

So Multi-head property can be viewed as regularizing to attend different subspace of representation, by preventing averaging overall attention score.

In summary, self-attention is just matrix multiplication that essentially does adjust the importance of each word in a sentence to make a better representation.

### Additional Model Capacity and Regularization

After retrieving self-attention applied representation from the previous step,
The `BertSelfOutput` module provides additional linear transformation with layer normalization and Dropout.
I think the role of this component is to provide additional model capacity and add Regularization.

## Our Task: Language Modeling
We use our model as language modeling.
Language modeling is an NLP task that predicts the next word from a given sequence of words.
The Transformer approach to language modeling is text classification, where calculate conditional probabilities of the next word in given previous words.

`BertModel` creates embedding and passes it to the encoder.
 The encoder provides a representation of a sentence,
 and also the whole sentence (accumulated hidden states from all layers) is passed to the pooler for the whole sentence by making representation for whole sentences.
 
 
`BertForLM` takes the last hidden state (since language modeling is a classification for the next word) and apply Dropout and add passes to the linear layer to get a conditional distribution result.
It calculates loss after inference is done.

 

### Data Preprocessing
First of all, it splits data with N number of the batch for each train, eval loop.


<center>
<img src="./assets/lm-data.png" width="750" />
</center>

The above figure shows the data prepared for training.
Each row in the tensor contains a batch size number of words (20 in our case).
As example shows, a single sentence is represented by multiple batches with the same column number.


### Train/Eval Loop

#### training function
- Set model to training mode
- Define criterion, optimizer, and scheduler
- Fetch data
- Pass to model
- Calculate loss
- Backpropagation
- Gradient clipping
- Optimizer stepping
- Loss calculation
- Logging (epoch number, batches, loss, ppl)

#### evaluate function
- Set model to evaluation mode
- No grad definition
- Fetch data
- Pass to model (deterministic process)
- Total loss calculation

### Train/Eval Result & Ablation Studies

- `train_eval.log` shows numbers. Since train data is small, the model is close to underfitting.
- In larger transformer layer, more underfitting occur, thus loss function gets higher

- 1layer transformer encoder for LM task
![train-valid-graph](./assets/train_val_graph.png)

- 12layer transformer encoder for LM task (underfitting due to large model capacity)
![train-valid-graph_12layer](./assets/train_val_graph_12layer.png)



### Model Option

A common configuration for constructing the bert model. Options include,
> https://huggingface.co/transformers/main_classes/configuration.html#transformers.PretrainedConfig

- `vocab_size` for word embedding
- `hidden_size` for model internal(hidden) representation size
- `num_hidden_layers` for number of transformer layer to transforms hidden representation
- `num_attention_heads` split output of feedforward layer for different attention heads
- `intermediate_size` size of intermediate layer after self attention
- `hidden_act` activation function for intermediate layer after self attention
- `hidden_dropout_prob` hidden layer dropout prob. for regularization
- `attention_probs_dropout_prob` self attention layer dropout prob. for regularization
- `max_position_embeddings` max size of positional embedding legnth
- `type_vocab_size` max size of type vocab length (...)
- `layer_norm_eps` layer norm option
- `output_attentions` option for output attention prob
- `output_hidden_states` option for output hidden state


## References & Resources

### On Implementations
- Transformers: State-of-the-art Natural Language Processing for Pytorch and TensorFlow 2.0.
> https://github.com/huggingface/transformers
- Sequence-to-Sequence Modeling with nn.Transformer and TorchText
> https://pytorch.org/tutorials/beginner/transformer_tutorial.html
- transformers/modeling_bert.py at a7d46a060930242cd1de7ead8821f6eeebb0cd06 · huggingface/transformers (GitHub)
> https://github.com/huggingface/transformers/blob/a7d46a060930242cd1de7ead8821f6eeebb0cd06/src/transformers/models/bert/modeling_bert.py
- Configuration - transformers 3.5.0 documentation 
> https://huggingface.co/transformers/main_classes/configuration.html#transformers.PretrainedConfig
- The Annotated Transformer
> https://nlp.seas.harvard.edu/2018/04/03/attention.html
- Tensor2Tensor Intro
> https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb

### On Concepts
- Inductive bias - Wikipedia
> https://en.wikipedia.org/wiki/Inductive_bias
- Directed acyclic graph - Wikipedia 
> https://en.wikipedia.org/wiki/Directed_acyclic_graph
- Transfer learning and Image classification using Keras on Kaggle kernels.
> https://towardsdatascience.com/transfer-learning-and-image-classification-using-keras-on-kaggle-kernels-c76d3b030649
- Attention? Attention!
> https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html#multi-head-self-attention
- Attention Is All You Need
> https://arxiv.org/abs/1706.03762
