# DenseNet implement using TF 2.0

## What is DenseNet?

<img width="600" src="https://hoya012.github.io/assets/img/densenet/1_comment.png" 
alt="Prunus" title="DenseNet Figure 1">

It is very simple, and powerful cnn architecture.

This papers main idea is this - Don't discard previous feature!
<br><br>
Traditional CNN extract image feature from CNN layers, but it doens't use the CNN layer's features for classification. <br>
In contrast, After [Resnet](https://arxiv.org/abs/1512.03385), many CNN Architecture using skip connection(Resnet) 
and elementwise sum([Googlenet](https://arxiv.org/abs/1409.4842)). <br>
<br>
This paper is one of them, but the difference of this architecture doesn't excecute any Calculation, <br>just concatenate the previous layer to next layer

If you want more information, <br>you should check here (https://arxiv.org/abs/1608.06993)

## How to excecute?
You need Tensorlofw 2.x and numpy

You can install Tensorflow and numpy below code 

```python
pip install tensorflow
```

If you want to use gpu,

```python
pip install tensorflow-gpu
```
And you also need numpy
```python
pip install numpy
```

And you clone this repository.
```python
git clone https://github.com/yw0nam/DenseNet
```
Finally run train.py

```python
cd DenseNet
python train.py
```

That's all!<br>
If you want some insight of this implementation, please look the model.ipynb
