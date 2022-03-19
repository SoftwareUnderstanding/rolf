# Photo Caption Generation using Global and Local Attention


## Video Presentation Links

Full - https://drive.google.com/file/d/1GNtassuNpYC2YWWuV6kx7b2mSb6SNKek/view?usp=sharing


## Hosted on:
[ec2-3-218-186-119.compute-1.amazonaws.com](ec2-3-218-186-119.compute-1.amazonaws.com)  
[ec2-54-205-156-188.compute-1.amazonaws.com](ec2-54-205-156-188.compute-1.amazonaws.com)

## TensorBoard Link:
https://tensorboard.dev/experiment/r7g8gQT5T2SkEXOs4gXYTQ/

## Author(s): 
Poornapragna Vadiraj, Varun Bhaseen, Mirsaeid Abolghasemi

Deep Learning CMPE 258 - Term Project - Professor Vijay Eranti - Spring 2020

## Introduction:

Attention mechanisms are broadly used in present image captioning encoder / decoder frameworks, where at each step a weighted average is generated on encoded vectors to direct the process of caption decoding. 
However, the decoder has no knowledge of whether or how well the vector being attended and the attention question being given are related, which may result in the decoder providing erroneous results. 
Image captioning, that is to say generating natural automatic descriptions of language images are useful for visually impaired images and for the quest of natural language related pictures. 
It is significantly more demanding than traditional vision tasks recognition of objects and classification of images for two guidelines. 
First, well formed structured output space natural language sentences are considerably more challenging than just a set of class labels to predict. 
Secondly, this dynamic output space enables a more thin understanding of the visual scenario, and therefore also a more informative one visual scene analysis to do well on this task.

## Issues with Traditional Methods of Image Captioning:

During generating the next word of the caption, this word is usually describing only a part of the image. 
Capturing the essence of the entire input image is unable
Using the full representation of the image “h” to help in the process of generating each word cannot produce different words for different parts of the image. 
Using Attention mechanism to solve these problems, by focusing on certain important features that may be present in the image.
Eg: HAARCascade

![alt text](https://github.com/saeedabi1/deep_learning_project/blob/master/pictures/pasted%20image%200.png?raw=true)


## Why Attention Mechanism?

The problem with this method is that, when the model attempts to generate the caption next word, that word usually only describes a part of the image. 
It is not capable of capturing the essence of the whole input image. 
Using the entire representation of image to condition the generation of each word can not generate different words effectively for different parts of the image. 
This is why an Attention mechanism can be helpful.

## Data

To address this problem, there are many open source datasets available, such as 
* Flickr 8k (containing 8k images)
* Flickr 30k (containing 30k images)
* MS COCO (containing 180k images), etc. 

But we have used the Flickr 8k dataset for this case study which you can access from here. 

* Training a model with a large number of images on a system that is not a very high end PC / Laptop may not be feasible either.

* This dataset contains 8000 images with 5 captions each (as we already saw in the Introduction section that an image can have multiple captions, all of which are relevant at the same time).

* These images are bifurcated as follows: Training Set — 6000 images, Dev Set — 1000 images, and Test Set — 1000 images.




## Defining VGG Model:

The Image Model (VGG-16) defining which pre-trained: 
* The code below generates an instance of the VGG16 model utilizing the Keras API. 
* If we don't already have these, this immediately installs the appropriate data.



![alt text](https://github.com/saeedabi1/deep_learning_project/blob/master/pictures/Screen%20Shot%202020-05-23%20at%205.11.34%20PM.png?raw=true)



* The VGG16 model was pre-trained for classification of photos on the ImageNet data-set. 
* The VGG16 model contains a convolutionary part and a full-connected (or dense) part that is used to classify the image.
* he whole VGG16 model is downloaded which is about 528 MB if include_top=True.
* Only the convolutional part of the VGG16 model is downloaded which is just 57 MB if include_top=False
* Using fully connected layers in this pre-trained model, therefore downloading the full model is needed.

![alt text](https://github.com/saeedabi1/deep_learning_project/blob/master/pictures/unnamed.png?raw=true)





![alt text](https://github.com/saeedabi1/deep_learning_project/blob/master/pictures/Screen%20Shot%202020-05-23%20at%205.07.17%20PM.png?raw=true)


![alt text](https://github.com/saeedabi1/deep_learning_project/blob/master/pictures/Screen%20Shot%202020-05-23%20at%205.07.27%20PM.png?raw=true)



![alt text](https://github.com/saeedabi1/deep_learning_project/blob/master/pictures/Screen%20Shot%202020-05-23%20at%205.07.42%20PM.png?raw=true)


![alt text](https://github.com/saeedabi1/deep_learning_project/blob/master/pictures/Screen%20Shot%202020-05-23%20at%205.07.57%20PM.png?raw=true)


## Training steps:

* The ENCODER output, hidden state(initialised to 0) and the DECODER input(which is the <start> token) are passed to the DECODER.
* The DECODER returns the predictions and the DECODER hidden state.
* The DECODER hidden state is then passed back into the model and the predictions are used to calculate the loss.
* While training, we use the Teacher Forcing technique, to decide the next input of the DECODER.
* Teacher Forcing is the technique where the target word is passed as the next input to the DECODER. 
* This technique helps to learn the correct sequence or correct statistical properties from the sequence, quickly.
* Final step is to calculate the Gradient and apply it to the optimizer and backpropagate.

## Testing Step:

* It is similar to training step, just that we do not update the gradients, and provide the predicted output as decoder input to next RNN cell at next time steps.
* Test step is required to find out whether the model built is overfitting or not.

## Sample Outputs

![alt text](https://github.com/poornaprag/deep_learning_project/blob/master/pictures/pic1.png?raw=true)

![alt text](https://github.com/poornaprag/deep_learning_project/blob/master/pictures/pic2.png?raw=true)

## TensorBoard Summary	

![alt text](https://github.com/saeedabi1/deep_learning_project/blob/master/pictures/Screen%20Shot%202020-05-23%20at%205.08.25%20PM.png?raw=true)



![alt text](https://github.com/saeedabi1/deep_learning_project/blob/master/pictures/Screen%20Shot%202020-05-23%20at%205.08.39%20PM.png?raw=true)


## Evaluation:

* The evaluate function is similar to the training loop
* Except we don’t use Teacher Forcing here. 
* The input to Decoder at each time step is its previous predictions, along with the hidden state and the ENCODER output.
* Few key points to remember while making predictions.
  * Stop predicting when the model predicts the end token.
  * Store the attention weights for every time step.


## Greedy Search:

* Maximum Likelihood Estimation (MLE) i.e. 
* Selecting that word which is most likely according to the model for the given input. 
* It’s also called as Greedy Search, as we greedily select the word with maximum probability.


## Beam Search:

* Taking top k predictions
* Feed them again in the model
* Sort them using the probabilities returned by the model. 
* So, the list will always contain the top k predictions. 
* Taking the one with the highest probability
* Going through it till we encounter <end> or reach the maximum caption length.


## BLEU Score:

* The BLEU measure to evaluate the result of the test set generated captions. 
* The BLEU is simply taking the fraction of n-grams in the predicted sentence that appears in the ground-truth.
* BLEU is a well-acknowledged metric to measure the similarity of one hypothesis sentence to multiple reference sentences. 
* Given a single hypothesis sentence and multiple reference sentences, it returns value between 0 and 1. 
* The metric close to 1 means that the two are very similar.


![alt text](https://github.com/saeedabi1/deep_learning_project/blob/master/pictures/Screen%20Shot%202020-05-23%20at%205.23.58%20PM.png?raw=true)



![alt text](https://github.com/saeedabi1/deep_learning_project/blob/master/pictures/Screen%20Shot%202020-05-23%20at%205.28.24%20PM.png?raw=true)





## Conclusion:

* This is an example where the distribution of the train and test sets will be very different and no Machine Learning model will deliver good performance in the world in such cases. 
* Overall, we have to admit that our simple first-cut model does a good job of producing captions for pictures, without any stringent hyper-parameter tuning.
* We have to recognize that the photos used for research have to be semantically linked to those used in model training. 
* For eg, if we train our model on the photos of cats, dogs, etc., we don't have to check it on airplane photographs, waterfalls, etc. 
* The result for Beam Search was generally found to be better than Greedy Approach.



## Lessons Learned

* We were getting very poor accuracy at first 
* The results only improved after we did the following:
  * Doing more hyperparameter tuning
    * learning rate, batch size
    * number of layers
    * number of units
    * dropout rate
    * batch normalisation
  * For tuning Method 1’s  Model architecture:
    * Adding Batch Normalization Layer
    * Dropouts
    * Early stopping


## Supplemental Material used and submitted

The supplementary material might include:
* Notebook with:
  * Training
  * Testing
  * Validation
* Dataset
* Checkpoint Files
* Source code of web app
* Video
* Tensorboard visualizations


# Related Work:

* Image captioning has been extensively studied recently with encoder-decoder versions[1, 2, 3, 4, 5] 
* A CNN processes the input image in its basic form to transform it into a vector representation which is used as the initial input for an RNN. Given the previous word, the RNN sequentially predicts the next word in the caption without limiting the temporal dependence to a fixed order, as in n-gram-based approaches. 
* The representation of the CNN image can be entered in different ways in the RNN.While some authors [6, 7] Use this only to calculate the initial RNN status, while others enter it in each RNN iteration [8, 9].


## References:

[1] Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould. Spice: Semantic propositional image caption evaluation. In ECCV, 2016.

[2] Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, and Lei Zhang. Bottom-up and top-down attention for image captioning and visual question answering. In CVPR, 2018.

[3] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

[4] Hedi Ben-younes, Remi Cadene, Matthieu Cord, and Nicolas Thome. Mutan: Multimodal tucker fusion for visual question answering. In ICCV, 2017.

[5] Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. Scheduled sampling for sequence prediction with recurrent neural networks. In NeurIPS, 2015.

[6] Long Chen, Hanwang Zhang, Jun Xiao, Liqiang Nie, Jian Shao, Wei Liu, and Tat-Seng Chua. Sca-cnn: Spatial and channel-wise attention in convolutional networks for image captioning. In CVPR, 2017.

[7] Yangyu Chen, Shuhui Wang, Weigang Zhang, and Qingming Huang. Less is more: Picking informative frames for video captioning. In ECCV, 2018.

[8] Marco Pedersoli and Thomas Lucas and Cordelia Schmid and Jakob Verbeek: Areas of Attention for Image Captioning, arXiv, 2016.

[9] Sen He, Wentong Liao, Hamed R. Tavakoli, Michael Yang, Bodo Rosenhahn, and Nicolas Pugeault: Image Captioning through Image Transformer, arXiv, 2020.


## Links we referred to:

* Local Attention : https://arxiv.org/pdf/1502.03044.pdf
* Global Attention : https://arxiv.org/pdf/1508.04025.pdf
* https://www.appliedaicourse.com/lecture/11/applied-machine-learning-online-course/4150/attention-models-in-deep-learning/8/module-8-neural-networks-computer-vision-and-deep-learning
* Tensorflow Blog: https://www.tensorflow.org/tutorials/text/image_captioning
* https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
* https://towardsdatascience.com/intuitive-understanding-of-attention-mechanism-in-deep-learning-6c9482aecf4f
* Neural Machine Translation(Research Paper):https://arxiv.org/pdf/1409.0473.pdf


