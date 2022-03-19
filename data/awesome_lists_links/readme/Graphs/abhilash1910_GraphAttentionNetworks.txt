# GraphAttentionNetworks


## A framework implementation of Graph Attention Networks :robot:

![img1](imgs/GAT.jpg)

[This package](https://pypi.org/project/GraphAttentionNetworks/0.1/) is used for extracting Graph Attention Embeddings and provides a framework for a Tensorflow Graph Attention Layer which can be used for knowledge graph /node base semantic tasks. It determines the pair wise embedding matrix for a higher order node representation and concatenates them with an attention weight. It then passes it through a leakyrelu activation for importance sampling and damps out negative effect of a node.It then applies a softmax layer for normalization of the attention results and determines the final output scores.The GraphAttentionBase.py script implements a Tensorflow/Keras Layer for the GAT which can be used and the GraphMultiheadAttention.py is used to extract GAT embeddings.

This is a TensorFlow 2 implementation of Graph Attention Networks for generating node embeddings for Knowledge Graphs as well as for implementing a keras layer for Multihead Graph Attention from the paper, [Graph Attention Networks (Veličković et al., ICLR 2018)](https://arxiv.org/abs/1710.10903).


## Dependencies

<a href="https://www.tensorflow.org/">Tensorflow</a>


<a href="https://networkx.org/">Networkx</a>


<a href="https://scipy.org/">scipy</a>


<a href="https://scikit-learn.org/stable/">sklearn</a>



## Usability

Installation is carried out using the pip command as follows:

```python
pip install GraphAttentionNetworks==0.1
```

This library is built with Tensorflow:

<img src="https://media.wired.com/photos/5955aeeead90646d424bb349/master/pass/google-tensor-flow-logo-black-S.jpg">

The steps for generating Graph Attention Embeddings requires import of [GraphMultiheadAttention.py](https://github.com/abhilash1910/GraphAttentionNetworks/blob/master/GraphAttentionNetworks/GraphMultiheadAttention.py) script. An example is shown in the [test_Script.py](https://github.com/abhilash1910/GraphAttentionNetworks/blob/master/test_script.py)

Create a function to read the input csv file. The input should contain atleast 2 columns - source and target(labels). And both should be in text format. These can include textual extracts and their corresponding labels. The graph is then created as a MultiDigraph from [networkx] with the target and source columns from the input csv file. While generating the embeddings, the extracts from the labels are also considered and can be used to determine which label is the closest to the provided source(input text). In the example below, the 'test_gat_embeddings' method shows this. The dataset chosen for this demonstration is [Google Quest QnA](https://www.kaggle.com/c/google-quest-challenge) and as such any dataset having a source and a label column(textual contents) can be used to generate the embeddings. The  method requires the ```get_gat_embeddings``` method.This method takes as parameters: hidden_units (denotes the hidden embedding dimension of the neural network), num_heads(number of attention heads), epochs (number of training iterations),num_layers(number of layers for the network),mode(defaults to averaging mode attention, for concatenation see ```GraphAttentionBase.py```), the dataframe along with the source and target labels. The model outputs a embedding matrix (no of entries, no of hidden dims) and the corresponding graph.The dimensions are internally reduced to suit the output of the GAT embeddings.


```python
def test_gat_embeddings():
    print("Testing for VanillaGCN embeddings having a source and target label")
    train_df=pd.read_csv("E:\\train_graph\\train.csv")
    source_label='question_body'
    target_label='category'
    print("Input parameters are hidden units , number of layers,subset (values of entries to be considered for embeddings),epochs ")
    hidden_units=32
    num_layers=4
    subset=34
    epochs=40
    num_heads=8
    mode='concat'
    gat_emb,gat_graph=gat.get_gat_embeddings(hidden_units,train_df,source_label,target_label,epochs,num_layers,num_heads,mode,subset)
    print(gat_emb.shape)
    return gat_emb,gat_graph

```

### Theory

<img src="https://dsgiitr.com/images/blogs/GAT/GCN_vs_GAT.jpg">


- Neural GCN Multiplication: In order to obtain sufficient expressive power to transform the input features into higher level features, atleast one learnable linear transformation is required. To that end, as an initial step, a shared linear transformation, parametrized by a weight matrix, W∈RF'×F , is applied to every node.

- Self-Attention Pointwise: We then compute a pair-wise un-normalized attention score between two neighbors. Here, it first concatenates the z embeddings of the two nodes, where || denotes concatenation, then takes a dot product of it with a learnable weight vector  and applies a LeakyReLU in the end. This form of attention is usually called additive attention, in contrast with the dot-product attention used for the Transformer model. We then perform self-attention on the nodes, a shared attentional mechanism a : RF'×RF'→R to compute attention coefficients 

- Softmax Aggregation: In this case we are applying a softmax kernel on the attention scores (normalized) and then multiplying it with the feature map. The aggregation map can be concatenation or avergaing.This is the case for multihead attention. If we perform multi-head attention on the final (prediction) layer of the network, concatenation is no longer sensible and instead, averaging is employed, and delay applying the final nonlinearity (usually a softmax or logistic sigmoid for classification problems). 

The [GraphAttenionBase.py](https://github.com/abhilash1910/GraphAttentionNetworks/blob/master/GraphAttentionNetworks/GraphAttentionBase.py) implements the core GAT Multihead algorithm with both concatenation and aggregation variation. The returned output is of dimensions -> [batch size, number of nodes, labels]

For GCN embeddings please refer to the repository:[GCN](https://github.com/abhilash1910/SpectralEmbeddings)

### Test logs:

```Testing for GAT embeddings having a source and target label
Input parameters are hidden units , number of layers,subset (values of entries to be considered for embeddings),epochs 
adj (39, 39)
shape of target (39, 5)
Model: "model_50"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
second (InputLayer)             [(None, 39)]         0                                            
__________________________________________________________________________________________________
embedding_69 (Embedding)        (None, 39, 39)       1521        second[0][0]                     
__________________________________________________________________________________________________
first (InputLayer)              [(None, 39)]         0                                            
__________________________________________________________________________________________________
multihead_attention_31 (Multihe (None, 5)            4720        embedding_69[0][0]               
                                                                 first[0][0]                      
==================================================================================================
Total params: 6,241
Trainable params: 4,720
Non-trainable params: 1,521
__________________________________________________________________________________________________
None
Fitting model with {hidden_units} units
Epoch 1/200
WARNING:tensorflow:Gradients do not exist for variables ['multihead_attention_31/graph_attention_269/kernel_W:0', 'multihead_attention_31/graph_attention_269/bias:0', 'multihead_attention_31/graph_attention_270/kernel_W:0', 'multihead_attention_31/graph_attention_270/bias:0', 'multihead_attention_31/graph_attention_271/kernel_W:0', 'multihead_attention_31/graph_attention_271/bias:0', 'multihead_attention_31/graph_attention_272/kernel_W:0', 'multihead_attention_31/graph_attention_272/bias:0', 'multihead_attention_31/graph_attention_273/kernel_W:0', 'multihead_attention_31/graph_attention_273/bias:0', 'multihead_attention_31/graph_attention_274/kernel_W:0', 'multihead_attention_31/graph_attention_274/bias:0', 'multihead_attention_31/graph_attention_275/kernel_W:0', 'multihead_attention_31/graph_attention_275/bias:0', 'multihead_attention_31/graph_attention_276/kernel_W:0', 'multihead_attention_31/graph_attention_276/bias:0'] when minimizing the loss.
WARNING:tensorflow:Gradients do not exist for variables ['multihead_attention_31/graph_attention_269/kernel_W:0', 'multihead_attention_31/graph_attention_269/bias:0', 'multihead_attention_31/graph_attention_270/kernel_W:0', 'multihead_attention_31/graph_attention_270/bias:0', 'multihead_attention_31/graph_attention_271/kernel_W:0', 'multihead_attention_31/graph_attention_271/bias:0', 'multihead_attention_31/graph_attention_272/kernel_W:0', 'multihead_attention_31/graph_attention_272/bias:0', 'multihead_attention_31/graph_attention_273/kernel_W:0', 'multihead_attention_31/graph_attention_273/bias:0', 'multihead_attention_31/graph_attention_274/kernel_W:0', 'multihead_attention_31/graph_attention_274/bias:0', 'multihead_attention_31/graph_attention_275/kernel_W:0', 'multihead_attention_31/graph_attention_275/bias:0', 'multihead_attention_31/graph_attention_276/kernel_W:0', 'multihead_attention_31/graph_attention_276/bias:0'] when minimizing the loss.
WARNING:tensorflow:Gradients do not exist for variables ['multihead_attention_31/graph_attention_269/kernel_W:0', 'multihead_attention_31/graph_attention_269/bias:0', 'multihead_attention_31/graph_attention_270/kernel_W:0', 'multihead_attention_31/graph_attention_270/bias:0', 'multihead_attention_31/graph_attention_271/kernel_W:0', 'multihead_attention_31/graph_attention_271/bias:0', 'multihead_attention_31/graph_attention_272/kernel_W:0', 'multihead_attention_31/graph_attention_272/bias:0', 'multihead_attention_31/graph_attention_273/kernel_W:0', 'multihead_attention_31/graph_attention_273/bias:0', 'multihead_attention_31/graph_attention_274/kernel_W:0', 'multihead_attention_31/graph_attention_274/bias:0', 'multihead_attention_31/graph_attention_275/kernel_W:0', 'multihead_attention_31/graph_attention_275/bias:0', 'multihead_attention_31/graph_attention_276/kernel_W:0', 'multihead_attention_31/graph_attention_276/bias:0'] when minimizing the loss.
WARNING:tensorflow:Gradients do not exist for variables ['multihead_attention_31/graph_attention_269/kernel_W:0', 'multihead_attention_31/graph_attention_269/bias:0', 'multihead_attention_31/graph_attention_270/kernel_W:0', 'multihead_attention_31/graph_attention_270/bias:0', 'multihead_attention_31/graph_attention_271/kernel_W:0', 'multihead_attention_31/graph_attention_271/bias:0', 'multihead_attention_31/graph_attention_272/kernel_W:0', 'multihead_attention_31/graph_attention_272/bias:0', 'multihead_attention_31/graph_attention_273/kernel_W:0', 'multihead_attention_31/graph_attention_273/bias:0', 'multihead_attention_31/graph_attention_274/kernel_W:0', 'multihead_attention_31/graph_attention_274/bias:0', 'multihead_attention_31/graph_attention_275/kernel_W:0', 'multihead_attention_31/graph_attention_275/bias:0', 'multihead_attention_31/graph_attention_276/kernel_W:0', 'multihead_attention_31/graph_attention_276/bias:0'] when minimizing the loss.
2/2 [==============================] - 0s 2ms/step - loss: 1.6039 - acc: 0.2308
Epoch 2/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6078 - acc: 0.2821
Epoch 3/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6069 - acc: 0.2308
Epoch 4/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6084 - acc: 0.2821
Epoch 5/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6045 - acc: 0.2051
Epoch 6/200
2/2 [==============================] - 0s 0s/step - loss: 1.6063 - acc: 0.2051
Epoch 7/200
2/2 [==============================] - 0s 0s/step - loss: 1.6048 - acc: 0.2564
Epoch 8/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6092 - acc: 0.2564
Epoch 9/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6084 - acc: 0.2821
Epoch 10/200
2/2 [==============================] - 0s 0s/step - loss: 1.6071 - acc: 0.2821
Epoch 11/200
2/2 [==============================] - 0s 0s/step - loss: 1.6067 - acc: 0.2821
Epoch 12/200
2/2 [==============================] - 0s 0s/step - loss: 1.6045 - acc: 0.2564
Epoch 13/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6067 - acc: 0.2821
Epoch 14/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6073 - acc: 0.3333
Epoch 15/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6076 - acc: 0.2821
Epoch 16/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6050 - acc: 0.2564
Epoch 17/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6056 - acc: 0.3077
Epoch 18/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6060 - acc: 0.2821
Epoch 19/200
2/2 [==============================] - 0s 0s/step - loss: 1.6058 - acc: 0.2308
Epoch 20/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6052 - acc: 0.2821
Epoch 21/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6039 - acc: 0.3077
Epoch 22/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6086 - acc: 0.2821
Epoch 23/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6049 - acc: 0.2821
Epoch 24/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6068 - acc: 0.2821
Epoch 25/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6050 - acc: 0.2821
Epoch 26/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6051 - acc: 0.3077
Epoch 27/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6069 - acc: 0.3077
Epoch 28/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6060 - acc: 0.3077
Epoch 29/200
2/2 [==============================] - 0s 0s/step - loss: 1.6046 - acc: 0.3077
Epoch 30/200
2/2 [==============================] - 0s 0s/step - loss: 1.6050 - acc: 0.3077
Epoch 31/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6034 - acc: 0.2821
Epoch 32/200
2/2 [==============================] - 0s 0s/step - loss: 1.6075 - acc: 0.3077
Epoch 33/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6055 - acc: 0.3077
Epoch 34/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6047 - acc: 0.3077
Epoch 35/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6007 - acc: 0.3333
Epoch 36/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6066 - acc: 0.2821
Epoch 37/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6048 - acc: 0.3077
Epoch 38/200
2/2 [==============================] - 0s 0s/step - loss: 1.6048 - acc: 0.2821
Epoch 39/200
2/2 [==============================] - 0s 0s/step - loss: 1.6044 - acc: 0.2821
Epoch 40/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6061 - acc: 0.3333
Epoch 41/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6036 - acc: 0.3333
Epoch 42/200
2/2 [==============================] - 0s 0s/step - loss: 1.6054 - acc: 0.3333
Epoch 43/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6074 - acc: 0.2821
Epoch 44/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6042 - acc: 0.3333
Epoch 45/200
2/2 [==============================] - 0s 0s/step - loss: 1.6033 - acc: 0.3077
Epoch 46/200
2/2 [==============================] - 0s 0s/step - loss: 1.6082 - acc: 0.2821
Epoch 47/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6079 - acc: 0.3077
Epoch 48/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6045 - acc: 0.2821
Epoch 49/200
2/2 [==============================] - 0s 0s/step - loss: 1.6041 - acc: 0.3077
Epoch 50/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6059 - acc: 0.3077
Epoch 51/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6038 - acc: 0.3333
Epoch 52/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6004 - acc: 0.3333
Epoch 53/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6056 - acc: 0.3077
Epoch 54/200
2/2 [==============================] - 0s 0s/step - loss: 1.6040 - acc: 0.2821
Epoch 55/200
2/2 [==============================] - 0s 0s/step - loss: 1.6050 - acc: 0.3333
Epoch 56/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6020 - acc: 0.3846
Epoch 57/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6013 - acc: 0.3333
Epoch 58/200
2/2 [==============================] - 0s 0s/step - loss: 1.6039 - acc: 0.3590
Epoch 59/200
2/2 [==============================] - 0s 0s/step - loss: 1.6019 - acc: 0.3590
Epoch 60/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5997 - acc: 0.3333
Epoch 61/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6026 - acc: 0.3846
Epoch 62/200
2/2 [==============================] - 0s 0s/step - loss: 1.6022 - acc: 0.3590
Epoch 63/200
2/2 [==============================] - 0s 0s/step - loss: 1.6020 - acc: 0.3590
Epoch 64/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6011 - acc: 0.3333
Epoch 65/200
2/2 [==============================] - 0s 0s/step - loss: 1.5982 - acc: 0.3846
Epoch 66/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6003 - acc: 0.3590
Epoch 67/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6015 - acc: 0.4103
Epoch 68/200
2/2 [==============================] - 0s 0s/step - loss: 1.6045 - acc: 0.3846
Epoch 69/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5975 - acc: 0.4103
Epoch 70/200
2/2 [==============================] - 0s 0s/step - loss: 1.5987 - acc: 0.3846
Epoch 71/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5999 - acc: 0.4103
Epoch 72/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6009 - acc: 0.4103
Epoch 73/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5986 - acc: 0.4103
Epoch 74/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6013 - acc: 0.4359
Epoch 75/200
2/2 [==============================] - 0s 0s/step - loss: 1.6026 - acc: 0.4103
Epoch 76/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5995 - acc: 0.4615
Epoch 77/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6012 - acc: 0.4359
Epoch 78/200
2/2 [==============================] - 0s 0s/step - loss: 1.6024 - acc: 0.4103
Epoch 79/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6027 - acc: 0.3846
Epoch 80/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6023 - acc: 0.4615
Epoch 81/200
2/2 [==============================] - 0s 0s/step - loss: 1.6014 - acc: 0.4615
Epoch 82/200
2/2 [==============================] - 0s 0s/step - loss: 1.5994 - acc: 0.4872
Epoch 83/200
2/2 [==============================] - 0s 0s/step - loss: 1.6013 - acc: 0.4872
Epoch 84/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6020 - acc: 0.4359
Epoch 85/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5979 - acc: 0.4103
Epoch 86/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6035 - acc: 0.4872
Epoch 87/200
2/2 [==============================] - 0s 0s/step - loss: 1.6046 - acc: 0.5128
Epoch 88/200
2/2 [==============================] - 0s 0s/step - loss: 1.6021 - acc: 0.4103
Epoch 89/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6028 - acc: 0.4615
Epoch 90/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6002 - acc: 0.4872
Epoch 91/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6001 - acc: 0.4615
Epoch 92/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6028 - acc: 0.4615
Epoch 93/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6005 - acc: 0.5385
Epoch 94/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5985 - acc: 0.5128
Epoch 95/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6020 - acc: 0.4872
Epoch 96/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5957 - acc: 0.4872
Epoch 97/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6019 - acc: 0.4872
Epoch 98/200
2/2 [==============================] - 0s 0s/step - loss: 1.6014 - acc: 0.5128
Epoch 99/200
2/2 [==============================] - 0s 0s/step - loss: 1.5995 - acc: 0.5128
Epoch 100/200
2/2 [==============================] - 0s 0s/step - loss: 1.6011 - acc: 0.5128
Epoch 101/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6000 - acc: 0.4872
Epoch 102/200
2/2 [==============================] - 0s 0s/step - loss: 1.6007 - acc: 0.5385
Epoch 103/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5975 - acc: 0.5385
Epoch 104/200
2/2 [==============================] - 0s 0s/step - loss: 1.5974 - acc: 0.5385
Epoch 105/200
2/2 [==============================] - 0s 0s/step - loss: 1.6020 - acc: 0.5128
Epoch 106/200
2/2 [==============================] - 0s 0s/step - loss: 1.5975 - acc: 0.5385
Epoch 107/200
2/2 [==============================] - 0s 0s/step - loss: 1.5993 - acc: 0.5128
Epoch 108/200
2/2 [==============================] - 0s 0s/step - loss: 1.5992 - acc: 0.5641
Epoch 109/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5991 - acc: 0.5385
Epoch 110/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5996 - acc: 0.5128
Epoch 111/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6018 - acc: 0.5641
Epoch 112/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6007 - acc: 0.5385
Epoch 113/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6006 - acc: 0.4872
Epoch 114/200
2/2 [==============================] - 0s 0s/step - loss: 1.5966 - acc: 0.5385
Epoch 115/200
2/2 [==============================] - 0s 0s/step - loss: 1.6022 - acc: 0.5385
Epoch 116/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5987 - acc: 0.5128
Epoch 117/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5985 - acc: 0.5128
Epoch 118/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5979 - acc: 0.5641
Epoch 119/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5981 - acc: 0.5128
Epoch 120/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5987 - acc: 0.5385
Epoch 121/200
2/2 [==============================] - 0s 0s/step - loss: 1.6007 - acc: 0.5385
Epoch 122/200
2/2 [==============================] - 0s 0s/step - loss: 1.5975 - acc: 0.5128
Epoch 123/200
2/2 [==============================] - 0s 0s/step - loss: 1.5985 - acc: 0.5128
Epoch 124/200
2/2 [==============================] - 0s 0s/step - loss: 1.5992 - acc: 0.5128
Epoch 125/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5964 - acc: 0.5385
Epoch 126/200
2/2 [==============================] - 0s 2ms/step - loss: 1.6006 - acc: 0.5385
Epoch 127/200
2/2 [==============================] - 0s 0s/step - loss: 1.6001 - acc: 0.5641
Epoch 128/200
2/2 [==============================] - 0s 0s/step - loss: 1.5963 - acc: 0.5641
Epoch 129/200
2/2 [==============================] - 0s 0s/step - loss: 1.5989 - acc: 0.5385
Epoch 130/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5962 - acc: 0.5641
Epoch 131/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5981 - acc: 0.5385
Epoch 132/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5953 - acc: 0.5385
Epoch 133/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5950 - acc: 0.5385
Epoch 134/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5981 - acc: 0.5641
Epoch 135/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5975 - acc: 0.5897
Epoch 136/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5927 - acc: 0.5641
Epoch 137/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5965 - acc: 0.5385
Epoch 138/200
2/2 [==============================] - 0s 0s/step - loss: 1.5960 - acc: 0.5385
Epoch 139/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5975 - acc: 0.5641
Epoch 140/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5975 - acc: 0.5897
Epoch 141/200
2/2 [==============================] - 0s 0s/step - loss: 1.5974 - acc: 0.5641
Epoch 142/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5989 - acc: 0.5897
Epoch 143/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5968 - acc: 0.5897
Epoch 144/200
2/2 [==============================] - 0s 0s/step - loss: 1.5980 - acc: 0.5641
Epoch 145/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5980 - acc: 0.6410
Epoch 146/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5969 - acc: 0.5897
Epoch 147/200
2/2 [==============================] - 0s 0s/step - loss: 1.5969 - acc: 0.6154
Epoch 148/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5932 - acc: 0.5897
Epoch 149/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5948 - acc: 0.5641
Epoch 150/200
2/2 [==============================] - 0s 4ms/step - loss: 1.6012 - acc: 0.5641
Epoch 151/200
2/2 [==============================] - 0s 0s/step - loss: 1.5964 - acc: 0.5897
Epoch 152/200
2/2 [==============================] - 0s 0s/step - loss: 1.5949 - acc: 0.5897
Epoch 153/200
2/2 [==============================] - 0s 0s/step - loss: 1.5972 - acc: 0.6154
Epoch 154/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5945 - acc: 0.6410
Epoch 155/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5951 - acc: 0.5641
Epoch 156/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5981 - acc: 0.6410
Epoch 157/200
2/2 [==============================] - 0s 0s/step - loss: 1.5950 - acc: 0.6154
Epoch 158/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5949 - acc: 0.6154
Epoch 159/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5954 - acc: 0.5641
Epoch 160/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5978 - acc: 0.5385
Epoch 161/200
2/2 [==============================] - 0s 0s/step - loss: 1.5940 - acc: 0.5897
Epoch 162/200
2/2 [==============================] - 0s 0s/step - loss: 1.5975 - acc: 0.6154
Epoch 163/200
2/2 [==============================] - 0s 0s/step - loss: 1.5949 - acc: 0.6154
Epoch 164/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5939 - acc: 0.5897
Epoch 165/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5966 - acc: 0.6667
Epoch 166/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5935 - acc: 0.6410
Epoch 167/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5921 - acc: 0.6154
Epoch 168/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5950 - acc: 0.6667
Epoch 169/200
2/2 [==============================] - 0s 0s/step - loss: 1.5964 - acc: 0.5897
Epoch 170/200
2/2 [==============================] - 0s 0s/step - loss: 1.5950 - acc: 0.6667
Epoch 171/200
2/2 [==============================] - 0s 0s/step - loss: 1.5891 - acc: 0.6667
Epoch 172/200
2/2 [==============================] - 0s 0s/step - loss: 1.5924 - acc: 0.6410
Epoch 173/200
2/2 [==============================] - 0s 0s/step - loss: 1.5906 - acc: 0.6923
Epoch 174/200
2/2 [==============================] - 0s 0s/step - loss: 1.5937 - acc: 0.6667
Epoch 175/200
2/2 [==============================] - 0s 0s/step - loss: 1.5901 - acc: 0.6410
Epoch 176/200
2/2 [==============================] - 0s 0s/step - loss: 1.5975 - acc: 0.6667
Epoch 177/200
2/2 [==============================] - 0s 0s/step - loss: 1.5887 - acc: 0.6410
Epoch 178/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5905 - acc: 0.6667
Epoch 179/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5928 - acc: 0.6410
Epoch 180/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5903 - acc: 0.6410
Epoch 181/200
2/2 [==============================] - 0s 0s/step - loss: 1.5949 - acc: 0.6410
Epoch 182/200
2/2 [==============================] - 0s 0s/step - loss: 1.5896 - acc: 0.6923
Epoch 183/200
2/2 [==============================] - 0s 0s/step - loss: 1.5926 - acc: 0.6667
Epoch 184/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5933 - acc: 0.7179
Epoch 185/200
2/2 [==============================] - 0s 0s/step - loss: 1.5917 - acc: 0.6667
Epoch 186/200
2/2 [==============================] - 0s 0s/step - loss: 1.5891 - acc: 0.6667
Epoch 187/200
2/2 [==============================] - 0s 0s/step - loss: 1.5903 - acc: 0.6667
Epoch 188/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5922 - acc: 0.6667
Epoch 189/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5956 - acc: 0.6667
Epoch 190/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5955 - acc: 0.6667
Epoch 191/200
2/2 [==============================] - 0s 0s/step - loss: 1.5943 - acc: 0.6667
Epoch 192/200
2/2 [==============================] - 0s 0s/step - loss: 1.5904 - acc: 0.6410
Epoch 193/200
2/2 [==============================] - 0s 0s/step - loss: 1.5880 - acc: 0.6667
Epoch 194/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5994 - acc: 0.6410
Epoch 195/200
2/2 [==============================] - 0s 0s/step - loss: 1.5935 - acc: 0.6667
Epoch 196/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5899 - acc: 0.6923
Epoch 197/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5879 - acc: 0.6923
Epoch 198/200
2/2 [==============================] - 0s 2ms/step - loss: 1.5942 - acc: 0.6410
Epoch 199/200
2/2 [==============================] - 0s 0s/step - loss: 1.5971 - acc: 0.6923
Epoch 200/200
2/2 [==============================] - 0s 0s/step - loss: 1.5896 - acc: 0.6923
Dimensions of embeddings (39, 5)
[[0.2012526  0.20259565 0.198657   0.19916654 0.1983282 ]
 [0.19945449 0.19912691 0.19899039 0.20362976 0.19879843]
 [0.19992991 0.19903784 0.20286745 0.20007017 0.19809467]
 [0.20213112 0.199936   0.19900537 0.20027836 0.1986492 ]
 [0.19776767 0.20560838 0.20104136 0.19821526 0.1973673 ]
 [0.20053644 0.1997569  0.19718741 0.20560017 0.19691905]
 [0.20018028 0.19936264 0.1996499  0.19978425 0.20102301]
 [0.19945793 0.19914173 0.20245528 0.20021063 0.19873443]
 [0.19884491 0.20046438 0.19927275 0.20058933 0.20082861]
 [0.19917706 0.20128284 0.19944142 0.19948685 0.20061186]
 [0.19849762 0.19817543 0.19879887 0.20052965 0.20399839]
 [0.19931518 0.2013433  0.199615   0.19973907 0.19998741]
 [0.19847561 0.20165245 0.19825539 0.1994863  0.20213024]
 [0.19950281 0.20215113 0.19978707 0.20000161 0.19855739]
 [0.19828378 0.1995312  0.19828445 0.19930516 0.2045954 ]
 [0.20060964 0.20382763 0.19792375 0.20018485 0.19745411]
 [0.19908005 0.19907176 0.20212533 0.19942604 0.2002968 ]
 [0.2061225  0.1978762  0.1984831  0.1989557  0.19856244]
 [0.19875005 0.19904377 0.19953431 0.19955306 0.20311877]
 [0.19788706 0.19793089 0.19901524 0.19893484 0.20623192]
 [0.20016678 0.19998111 0.19964387 0.20063668 0.19957155]
 [0.19831264 0.19981474 0.20413318 0.19969617 0.19804329]
 [0.20088041 0.19825016 0.19835961 0.2013865  0.20112331]
 [0.20049077 0.20007299 0.19954118 0.1995979  0.20029715]
 [0.20174277 0.19962949 0.20002913 0.19979118 0.1988074 ]
 [0.20026413 0.19844735 0.20144895 0.20046745 0.19937217]
 [0.19872567 0.19828579 0.20021453 0.19848432 0.20428972]
 [0.19901909 0.2013395  0.20159248 0.19940722 0.19864176]
 [0.20733383 0.19861382 0.19859011 0.19829814 0.19716412]
 [0.19899431 0.20001376 0.19994445 0.19993514 0.20111232]
 [0.1980907  0.19873632 0.20662917 0.19852386 0.19801992]
 [0.20076685 0.19984347 0.19948737 0.20049627 0.1994061 ]
 [0.19691971 0.19762805 0.20314892 0.20342152 0.19888185]
 [0.2024308  0.19443996 0.20389585 0.1996117  0.19962174]
 [0.19936486 0.19557709 0.20349811 0.20444478 0.19711518]
 [0.19996345 0.22908325 0.1896215  0.18164895 0.19968286]
 [0.19765937 0.19925259 0.20080289 0.19597957 0.2063056 ]
 [0.2029202  0.19314696 0.20042692 0.19949025 0.20401572]
 [0.1998764  0.19227904 0.19804887 0.21609515 0.1937005 ]]
(39, 5)
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT
