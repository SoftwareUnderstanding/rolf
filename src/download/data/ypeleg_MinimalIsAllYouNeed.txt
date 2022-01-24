
<p align="center">
⚠️ Experimental and unstable yet. Still in development. ⚠️ 
</p>



## Minimal Is All You Need

Minimalist Transformers In [Keras](http://keras.io) that support sklearn's .fit .predict .

-----
##### What if you could just use transformers in keras without clutter...

```python
    from minimal_is_all_you_need import Bert, GPT_2, XLNet, ELMo, GPT,  Transformer, TransformerXL, the_loss_of_bert
```

##### And then simply: 

```python
    model = Bert()
    model.compile('adam', loss=[the_loss_of_bert(0.1), 'binary_crossentropy'])
    model.fit(X, Y)
```    

#### No games. 
#### No tricks.
#### No bloat.

-----

Have you ever wanted to work with transformers but just got drowned in ocean of models where none just did what you wanted?
Yeah.. Me Too..
Introducing "Minimal Is All You Need": Minimal Is All You Need is a Python library implementing nuts and bolts, for building Transformers models using [Keras](http://keras.io).

The library supports:
* Universal Transformer
* Bert
* ELMo
* GPT
* GPT-2
* TransformerXL
* XLNet
* positional encoding and embeddings,
* attention masking,
* memory-compressed attention,
* ACT (adaptive computation time),

### Examples:

### Bert
```python
    model = Bert()
    model.compile('adam', loss=[the_loss_of_bert(0.1), 'binary_crossentropy'])
```
<p align="center">
  <img src="https://github.com/ypeleg/MinimalIsAllYouNeed/blob/master/resources/Bert.png?raw=true" width="300">
</p>

### XLNet
```python
    model = XLNet()
    model.compile('adam', loss='sparse_categorical_crossentropy')
```
<p align="center">
  <img src="https://github.com/ypeleg/MinimalIsAllYouNeed/blob/master/resources/XLNet.png?raw=true" width="300">
</p>

### GPT-2
```python
    model = GPT_2()
    model.compile('adam', loss='sparse_categorical_crossentropy')
```
<p align="center">
  <img src="https://github.com/ypeleg/MinimalIsAllYouNeed/blob/master/resources/GPT_2.png?raw=true" width="300" height="1000">
</p>

### ELMo
```python
    model = ELMo()
    model.compile('adagrad', loss='sparse_categorical_crossentropy')
```
<p align="center">
  <img src="https://github.com/ypeleg/MinimalIsAllYouNeed/blob/master/resources/ELMo.png?raw=true" width="300">
</p>

### TransformerXL
```python
    model = TransformerXL()
    model.compile('adam', loss='sparse_categorical_crossentropy')
```
<p align="center">
  <img src="https://github.com/ypeleg/MinimalIsAllYouNeed/blob/master/resources/TransformerXL.png?raw=true" width="300">
</p>

### GPT
```python
    model = GPT()
    model.compile('adam', loss='sparse_categorical_crossentropy')
```
<p align="center">
  <img src="https://github.com/ypeleg/MinimalIsAllYouNeed/blob/master/resources/GPT.png?raw=true" width="300">
</p>

### Universal Transformer
```python
    model = Transformer()
    model.compile('adam', loss='sparse_categorical_crossentropy')
```
<p align="center">
  <img src="https://github.com/ypeleg/MinimalIsAllYouNeed/blob/master/resources/Transformer.png?raw=true" width="300">
</p>

It also allows you to piece together a multi-step Transformer model in a flexible way, for example: (Credit: Zhao HG)

```python
transformer_block = TransformerBlock( name='transformer', num_heads=8, residual_dropout=0.1, attention_dropout=0.1, use_masking=True)
add_coordinate_embedding = TransformerCoordinateEmbedding(transformer_depth, name='coordinate_embedding')
    
output = transformer_input # shape: (<batch size>, <sequence length>, <input size>)
for step in range(transformer_depth):
    output = transformer_block(add_coordinate_embedding(output, step=step))
```


All pieces of the model (like self-attention, activation function, layer normalization) are available as Keras layers, so, if necessary,
you can build your version of Transformer, by re-arranging them differently or replacing some of them.

The (Universal) Transformer is a deep learning architecture described in arguably one of the most impressive DL papers of 2017 and 2018:
the "[Attention is all you need][1]" and the "[Universal Transformers][2]"
by Google Research and Google Brain teams.

The authors brought the idea of recurrent multi-head self-attention,
which has inspired a big wave of new research models that keep coming ever since.
These models demonstrate new state-of-the-art results in various NLP tasks,
including translation, parsing, question answering, and even some algorithmic tasks.

Installation
------------
To install the library you need to clone the repository

    pip install minimal_is_all_you_need


### References 

[1]: https://arxiv.org/abs/1706.03762 Attention Is All You Need

[2]: https://arxiv.org/abs/1807.03819 Universal Transformers

[3]: https://arxiv.org/abs/1810.04805 BERT: Pre-training of Deep Bidirectional Transformers for


###  
### Important Note: 
#### Note: Not everything in this repository was made by myself. Parts of the code here and there were found online. 
#### I tried to give credict whenever this happened but mistakes might still happen!. 
#### If by any chance some authors didn't get their credit and you know it: PM ME! I will fix it ASAP!
#### Cheers ^^ 

