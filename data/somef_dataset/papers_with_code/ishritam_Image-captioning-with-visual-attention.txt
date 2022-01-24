# Image-captioning-with-visual-attention
To build networks capable of perceiving contextual subtleties in images, to relate observations to both the scene and the real world, and to output succinct and accurate image descriptions; all tasks that we as people can do almost effortlessly.

Deep Learning is a very rampant field right now – with so many applications coming out day by day. In this case study, I have made an Image Captioning refers to the process of generating textual description from an image – based on the objects and actions in the image. For example:
![image](https://user-images.githubusercontent.com/40149802/70611367-146bc500-1c2b-11ea-9941-75fa7366655e.png)

## Problem Statemtent
Image captioning is an interesting problem, where we can learn both computer vision techniques and natural language processing techniques. In case study I have followed Show, Attend and Tell: Neural Image Caption Generation with Visual Attention and create an image caption generation model using Flicker 8K data. This model takes a single image as input and output the caption to this image and read that predicted caption.

## Dependencies
* Python 3
* Tensorflow 2.0
* gtts

## Business Objectives and Constraints
* Predict a correct caption as per the input image.
* Incorrect caption could impact the negative impression on user.
* No strict latency constraints.

## Data Overview
Flilckr8K contains 8,000 images that are each paired with five different captions which provide clear descriptions of the salient entities and events. The images were chosen from six different Flickr groups, and tend not to contain any well-known people or locations, but were manually selected to depict a variety of scenes and situations.

## Sources:

* http://academictorrents.com/details/9dea07ba660a722ae1008c4c8afdd303b6f6e53b
* https://forms.illinois.edu/sec/1713398
> Flickr8k_Dataset: Contains all the images

> Flickr8k_text: Contains all the captions

## Mapping the real-world problem to a Deep Learning Problem
To accomplish this, we'll use an attention-based model, which enables us to see what parts of the image the model focuses on as it generates a caption.
“**Show, Attend and Tell: Neural Image Caption Generation with Visual Attention**” by Xu et al. (2015) — the first paper, to our knowledge, that introduced the concept of attention into image captioning. The work takes inspiration from attention’s application in other sequence and image recognition problems.
![image](https://user-images.githubusercontent.com/40149802/70611854-dd49e380-1c2b-11ea-9890-cdcb691d11ff.png)

## Key Performance Indicator (KPI)
As per the [Research paper](https://www.aclweb.org/anthology/P02-1040.pdf):

The primary programming task for a BLEU implementor is to compare n-grams of the candidate with the n-grams of the reference translation and count the number of matches. These matches are position-independent. The more the matches, the better the candidate translation is. 

BLEU is a well-acknowledged metric to measure the similarly of one hypothesis sentence to multiple reference sentences.
Given a single hypothesis sentence and multiple reference sentences, it returns value between 0 and 1.

The metric close to 1 means that the two are very similar. 
The metric was introduced in 2002 BLEU: a Method for Automatic Evaluation of Machine Translation. Although there are many problems in this metric, for example grammatical correctness are not taken into account, BLEU is very well accepted partly because it is easy to calculate.

* Higher the score better the quality of caption

# References:

* Tensorflow Tutorials: https://www.tensorflow.org/tutorials/text/image_captioning
* CS231n: Convolutional Neural Networks for Visual Recognition by Andrej Karpathy: https://www.youtube.com/watch?v=NfnWJUyUJYU&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC
* Show, Attend and Tell: Neural Image Caption
Generation with Visual Attention: https://arxiv.org/pdf/1502.03044.pdf
* Seq to seq model by Andrew Ng: https://www.youtube.com/watch?v=Q8ys8YnDRXM&list=PL1w8k37X_6L_s4ncq-swTBvKDWnRSrinI
* Attention Is All You Need: https://arxiv.org/pdf/1706.03762v5.pdf

* Show, Attend and Tell Paper presentation: https://www.youtube.com/watch?v=ENVGHs3yw7k&t=454s

* https://distill.pub/2016/augmented-rnns/
* https://fairyonice.github.io/Develop_an_image_captioning_deep_learning_model_using_Flickr_8K_data.html#Add-start-and-end-sequence-tokens

* BLEU score: https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
