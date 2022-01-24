# Glossary 

In this document, I will try to define some notions in a few words as I stambled upon a word. So don't expect to find something well structured here. 

## Attention mechanism ([Bahdanau et al., 2015])(https://arxiv.org/pdf/1409.0473.pdf)

An attention mechanism aims at telling to the neural network how much it is supposed to pay attention to some part of the input data. This allow to take less attention to noise and only used important features for the final task.

For example in this sentence, we can clearly understand the concept :
<img src="images/ex_sentence_att.PNG">

Attention has been introduced in the NLP domain. The aim was to alleviate the seq2seq problem of the fixed size of the context vector making impossible to memorize first parts of a long sequence.

The secret sauce invented by attention is to create shortcuts between the context vector and the entire source input. The weights of these shortcut connections are customizable for each output element.


Afterwards, attention was extended to computer vision field. 

## Transformer 
Transformer is a type of attention mechanism


## Geodesic 
"In differential geometry, a geodesic (/ˌdʒiːəˈdɛsɪk, ˌdʒiːoʊ-, -ˈdiː-, -zɪk/[1][2]) is a generalization of the notion of a "straight line" to "curved spaces"." [wikipedia](https://en.wikipedia.org/wiki/Geodesic)

## Huber loss

The huber loss function is used in robust regression because it is less sensitive to outliers in data than the squarred error loss.

<img src='images/huber_loss_eq.PNG'> 

The huber loss is quadratic for small values and linear for large values.

<img src='images/huber_loss_curve.PNG'>
