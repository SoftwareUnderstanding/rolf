# Exercises :)
This repo gathers various exercises on ML problems. Some of them are still active projects, others are completed. 

## Collaborative filtering exercise

This comes from ```Computational Intelligence Lab``` class ```@ETH``` during 2020 spring semester. <br>
The exercise explores basic matrix completion/factorization techniques and exposes their limitation. 
Also, you'll find an hand-made implementation of SGD.

## PLSA exercise
This comes from ```Computational Intelligence Lab``` class ```@ETH``` during 2020 spring semester. <br>
The exercise explores Probabilistic latent semantic analysis as presented in the paper https://arxiv.org/pdf/1301.6705.pdf. 
The goal of the exercise is to implement PLSA by using the EM algorithm, and to apply it to the Associated Press corpus dataset.

## Sentiment analysis exercise -> SA repo
This comes from ```Computational Intelligence Lab``` class ```@ETH``` during 2020 spring semester. <br>
This is an exploratory notebook used for the (ongoing) *Kaggle Competition* https://www.kaggle.com/c/cil-text-classification-2020/data.

Up until now in the notebook I have implemented:
- vocabulary extraction 
- **GloVe** embedding training with **Stochastic Gradient Descent**
- training pipeline (using the learned embeddings to train a sentiment classifier)
- predictions pipeline (using the learned embeddings and the trained classifier to predict tweets sentiment)

**Update 12.04.20**:<br>
The project moved to a dedicated repo. You can see all the changes there. 

*Note*: to be able run this notebook you need to have the *.zip* datasets in your current working file system, or in your *Google Drive* folder.

## Writing like Dante

This comes from ```Machine Perception``` class ```@ETH``` during 2020 spring semester. <br>
The inspiration from this notebook was drawn from [this beautiful blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).
The idea is to train an LSTM on *character-level text prediction* using a single .txt file. I chose a text from the [Gutenberg](http://www.gutenberg.org/) repository, namely *la Divina Commedia* , by *Dante Alighieri*. 
<br> Maybe what is most exciting about this work is perfectly synthesized by *Andrej Karpathy* (the author of the blog post): 
 > There’s something magical about Recurrent Neural Networks (RNNs). [...] Sometimes the ratio of how simple your model is to the quality of the results you get out of it blows past your expectations, and this was one of those times. 
 
Here's a preview of the results: 
> E io a lui: «Dor, fosse, tra miggia,<br>
  non li oche giaco, se Cercoresto daicra<br>
  feruto alcundon par de' Fuorchi sonno,<br>
  me steso le sue cadëa in avro si tecco.<br><br>
  Tattir si poccia a corpo par savilsa;<br>
  e ch'io forse di quanto a terra soscruto,<br>
  mi fu maca scricciatà il focose<br>
  al figuboro e deggea feder puoso.<br><br>
  Per convien la bercine dol posova<br>
  de l'un cheggio, e a le piaggi traddume.<br>


## Image compression by clustering
<img src="https://github.com/GiuliaLanzillotta/exercises/blob/master/compressions.jpg">

This comes from ```Computational Intelligence Lab``` class ```@ETH``` during 2020 spring semester. <br>
The goal of this notebook was to explore different clustering algorithms in the setting of *Image Compression*. <br>
Above you can see the result obtained with K-means for different values of k (the number of clusters). 

## Dijkstra’s algorithm (in fieri)

Here the goal is to implement a Monte Carlo simulation that calculates the average shortest path in a graph. The shortest path algorithm will be Dijkstra’s.

## Tweet Generator (in fieri)
Implementation of **Generating sentences from a continuous space paper**<br>
Here's a link to the [paper](https://arxiv.org/pdf/1511.06349v4.pdf).

> ### The goal 
> What I want to explore here is the expression of sentiment in generative models. <br>
> The dataset consists of two different samples of tweets, one with positive sentiment and one with negative sentiment. <br>
> The goal is to train two generators on the two sets separetly and analyse the qualitative differences.

*Note*: to be able run this notebook you need to have the *.zip* datasets in your current working file system, or in your *Google Drive* folder.

## Turning my sister into an old painting (in fieri)
This notebook will explore the magic of GANs. <br>
We are going to refer to a particular GAN architecture : **the Cycle GAN**. Here's a [link to the paper](https://arxiv.org/pdf/1703.10593.pdf) for the more curious. 

>  The goal? Taking a picture of my beautiful sister and turn it into a painting, to see how she would have looked like a few centuries ago. 


## Signal processing 
<img src="https://github.com/GiuliaLanzillotta/exercises/blob/master/pics/sp1.png">
 

<div>
 
This exercise draws inspiration from a pair of lectures in the ```Computational Intelligence Lab``` class ```@ETH``` during 2020 spring semester. The topic is *signal processing* in the realm of *lossy data compression*. <br>
The notebook goes through some maths and applies it to 1D signal first, and to images in the end. 
In the plot below you can see the results of image compression using FFT transform (center) and DCT transform (right).

</div>
 
<div align="center">
<img src="https://github.com/GiuliaLanzillotta/exercises/blob/master/pics/sp3.png" width="250" >
<img src="https://github.com/GiuliaLanzillotta/exercises/blob/master/pics/sp4.png" width="250" >
<img src="https://github.com/GiuliaLanzillotta/exercises/blob/master/pics/sp5.png" width="250">
</div>



