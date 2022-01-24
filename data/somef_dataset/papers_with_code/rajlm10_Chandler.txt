# Chandler

## Motivation
Sarcasm is tough to detect sometimes even at a human level making it even tougher for machines to detect. I wanted to incorporate state of the art techniques to address this problem. The code in this repo is solely in tensorflow and the models have been created from scratch to understand the workings of an Encoder thouroughly. The implementation is based on the paper titled. **Attention is all you need** (Vaswani et al). [Here](https://arxiv.org/abs/1706.03762) is the link to the original paper. Some insights were also derived from the paper titled **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** (Google AI Language). This paper can be found [here](https://arxiv.org/pdf/1810.04805.pdf).
I will try and expain each section of the code here and also provide excerpts from the paper/some medium blogs which helped me understand the code better.
*Feel free to navigate to any section you want through the navbar below.*

## Quick Note
For those of you who solely want to achieve high performance I would recommend using a pre-trained BERT model from Hugging Face and fine-tuning it on the dataset I have used. However this repo aims at understanding the Encoder model better from scratch. Also note that the final model in production is very light (**29 MB** Weights + **7MB** Tokenizer) whereas BERT or even DISTILBERT for that matter consumes much more space but at the same time is much more robust since it has more Encoder Layers, Feed Forward Network Units and is trained on huge data using different objectives and has a larger vocabulary size. The BERT paper linked above will tell you more about this.

## Details about the size (Feel free to skip this if you don't want the mathematical details)
For those of you wondering about the size. The model used here has got an embedding dimesnion of **256** has been used opposed to BERT's **768**. The embedding dimension has been decoupled from the hidden_dims (**512**) inside the model (and its Dense layers). So the embedding dimensions are **1/3** of those from **BERT** and hidden dims are **2/3** of **BERT**. Also note that the original paper introduces an intermediate dense projection of dimesnion **intermidiate_dims=3072**. In the original BERT workflow **768** dims from the attention output get projected to **3072** dims which then get projected to **768** dims again. This intermediate projection gives the transformer tremendous power. In my workflow the **256 dims from the attention layer get projected to 512 dims and then again to 256 dims**
which reduced alot of power. Since I am not using this model of the intention of transfer learning on other downstream tasks and this model is task-specific to detecting sarcasm, it does the job, but to use the encoder in its true sense (few-shot learning on down-stream tasks) we would need much more power (in terms of dimensions). So to sum it up basically we reduce the **BERT_Vocab_Size (30000) X 768 matrix to a Small_Vocab_size X 256 matrix. We reduce the 768 X 3072 and 3072 X 768 matrices to 512 X 256 and 256 X 512 matrix for EACH ENCODER's FFN layers. And furthermore we have only 6 Encoders compared to BERT's 12. Hence the massive reduction.**

_ALBERT uses this technique of decoupling the embedding_dims and hidden_dims in the encoder._


## TODO
Compare performance drop by **sharing weights across all 6 ENCODERs (INSPIRED FROM ALBERT)**

## Jump To

* <a id="jumpto"></a> [Dataset](#dataset-)
* <a id="jumpto"></a> [Self-Attention](#self-attention-)
* <a id="jumpto"></a> [Multi-Headed-Attention](#multi-headed-attention-)
* <a id="jumpto"></a> [Positional-Encodings](#positional-encodings-)
* <a id="jumpto"></a> [Encoder](#encoder-)
* <a id="jumpto"></a> [Adapting the Encoder to a classification problem](#adapting-the-encoder-to-a-classification-problem-)
* <a id="jumpto"></a> [Variations](#variations-)
* <a id="jumpto"></a> [Training](#training-)
* <a id="jumpto"></a> [Comparision](#comparision-)
* <a id="jumpto"></a> [Model Selection](#model-selection-)
* <a id="jumpto"></a> [Results](#results-)
* <a id="jumpto"></a> [Usage and Tips](#usage-and-tips-)
* <a id="jumpto"></a> [Retrospection](#retrospection-)
* <a id="jumpto"></a> [References](#references-)

# Dataset [`↩`](#jumpto)
The dataset contains news headline from two news website. [*TheOnion*](https://www.theonion.com/) which aims at producing sarcastic versions of current events and   non-sarcastic news headlines from [*HuffPost*](https://www.huffingtonpost.com/).

The dataset can be found in this [repo](https://github.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection).
Here are some stats from the official repo
| Statistic/Dataset                              | Headlines | 
|------------------------------------------------|-----------|
| # Records                                      | 28,619    | 
| # Sarcastic records                            | 13,635    |   
| # Non-sarcastic records                        | 14,984    | 
| % of pre-trained word embeddings not available | 23.35     |   

Citation
```
@article{misra2019sarcasm,
  title={Sarcasm Detection using Hybrid Neural Network},
  author={Misra, Rishabh and Arora, Prahal},
  journal={arXiv preprint arXiv:1908.07414},
  year={2019}
}

@book{book,
author = {Misra, Rishabh and Grover, Jigyasa},
year = {2021},
month = {01},
pages = {},
title = {Sculpting Data for ML: The first act of Machine Learning},
isbn = {978-0-578-83125-1}
}
```

# Self-Attention [`↩`](#jumpto)
*Self-attention is a sequence-to-sequence operation: a sequence of vectors goes in, and a sequence of vectors comes out. Let’s call the input vectors x1, x2,…, xt and the corresponding output vectors y1, y2,…, yt. The vectors all have dimension k. To produce output vector yi, the self attention operation simply takes a weighted average over all the input vectors, the simplest option is the dot product.* 

**The Query, The Value and The Key**

Every input vector is used in three different ways in the self-attention mechanism: the Query, the Key and the Value.
In every role, it is compared to the other vectors to get its own output yi(Query), to get the j-th output yj(Key) and to compute each output vector once the weights have been established (Value).
To obtain this roles, we need three weight matrices of dimensions d_model x d_model and compute three linear transformation for each xi:
These three matrices are usually known as K, Q and V, three learnable weight layers that are applied to the same encoded input. Consequently, as each of these three matrices come from the same input, we can apply the attention mechanism of the input vector with itself, a “self-attention”.

*Here is the equation from the paper. This is called *scaled dot-product attention*

![](/images/self-attn.png)

# Multi-Headed-Attention [`↩`](#jumpto)

We basically parallelize the scaled dot product attention discussed in the previous section. Instead of having one attention operation do this n times , n being the number of attention heads. Each head gets a part (equally) of the query, key and value. In the previous description the attention scores are focused on the whole sentence at a time, this would produce the same results even if two sentences contain the same words in a different order. Instead, we would like to attend to different segments of the words. We can give the self attention greater power of discrimination, by combining several self attention heads, dividing the words vectors into a fixed number (h, number of heads) of chunks, and then self-attention is applied on the corresponding chunks, using Q, K and V sub-matrices.[http://peterbloem.nl/blog/transformers]. Later attention scores from these various heads are again concatenated and multiplied by a set of trainable weights.

*This image from the transformer paper helps understand multi-headed-attention better*


![](/images/multi-head.png)

# Positional-Encodings [`↩`](#jumpto)

The problem with Encoders is that they are fed a sequence all at once rather than sequentially. This is why they are also called "Bi-Directional". Now a  problem arises that if we only use word embeddings we are not taking into account the position of each word. In summary any two sentences with the same words in different orders will be viewed the same by our encoder. To make the encoder aware about the position of these words we need an additional embedding which is concatenated with the word embeddings. One approach would be to use a position embedding, similar to word embedding, coding every known position with a vector. “It would requiere sentences of all accepted positions during the training loop but positional encoding allow the model to extrapolate to sequence lengths longer than the ones encountered during training”. Moreover while training we would need sentences of all lengths (upto maxlen) to make sure the positional embedding layer learns. However the paper addresses this by using a sinusodial function. (The code contains more insights to the equation)

The equation from the Transformer paper:

![](/images/position.png)


# Encoder [`↩`](#jumpto)

The best resource to learn about the Encoder is surely Jay Alammar's [The illustrated transformer](http://jalammar.github.io/illustrated-transformer/). I would strongly suggest new learners to go through the entire blog. It also illustrates the Decoder which we won't be using for our project.

In essence each encoder layer consists of multi-headed attention applied to inputs which are embedded and passed through the positional encoding layer. The next phase after multi-headed attention is the dropout regularization followed by a residual layer (inputs+multi-headed-attn) followed by Feed Forward Networks (I have used 512 units) It also has a reLU activation , a dropout layer and finally one more residual layer. However this encoder has not been adapted for sequence classifcation tasks yet. I will talk about this in the next section.

This makes up one layer but in practice multiple such encoder layers (stacked up) are used to make up one whole encoder. I have used 6 for this task.

*The Encoder from Jay Alammar's blog:*

![](/images/encoder.png)

# Adapting the Encoder to a classification problem [`↩`](#jumpto)

The encoder in the paper was used for a machine translation task. It is important to know this because we will be using the encoder for a binary classification task. The last layer from the encoder outputs a tensor of shape **BATCH_SIZE X MAX_SEQ_LEN X EMBEDDING_DIM** . In essence this can be considered a hidden state for each token since we have MAX_SEQ_LEN of these tokens. However to use it as a sequence classifier we need to reduce this tensor to a shape **BATCH_SIZE X EMBEDDING_DIM** before passing it through a Dense layer (containing n target classes) . This [stackoverflow](https://stackoverflow.com/questions/58123393/how-to-use-transformers-for-text-classification) answer helps offer some choices. Here are some choices to play around with 

1. Average all the hidden states along the axis **MAX_SEQ_LEN**
2.  According to the **BERT** paper prepend a **[CLS]** token to each sentence and use the hidden state of this during classification. Note that using a single token like this requires extensive training as the token must learn as much from context as possible.
3.  Use a Conv2D layer and then a Flatten Layer to make sure the dimensions are **BATCH_SIZE X EMBEDDING_DIM**
4.  Use an LSTM layer before the final Dense layer

I will be using techniques listed in 1,3 and 4 to create models and will compare them. The diagrams for each are listed in the next section.

# Variations [`↩`](#jumpto)

Let us go through each of the model variations listed in the above section visually.

**1. Average all the hidden states along the axis *MAX_SEQ_LEN***


![](/images/reduce-mean.png)

**2. Use an LSTM layer (128) before the final Dense layer**


![](/images/lstm.png)

**3. Use a Conv2D layer and then a Flatten Layer to make sure the dimensions are *BATCH_SIZE X EMBEDDING_DIM***

*The filter size used was 3X3 and 'same' padding was used along with a 'reLU' activation.*

![](/images/conv.png)

# Training [`↩`](#jumpto)

All the three models were trained on a GPU on Google Colab. All of them were trained against the Binary Cross Entropy loss function. According to the transformer paper **Adam** was used as an optimizer with a variable learning rate varying according to the model dimensions as per this equation below.

![](/images/training.png)

All of them were trained for 5 epochs on the GPU with a batch size of 128. By the end these were the training accuracies and losses for each model.

| **Model**           	| **Loss**  	| **Training Accuracy** 	|
|-----------------	|--------	|-------------------	|
| Encoder         	| 0.2304 	| 90.18 %           	|
| Encoder + LSTM  	| 0.2303 	| 90.48 %           	|
| Encoder + Conv2D   	| 0.2489 	| 89.48 %           	|

# Comparision [`↩`](#jumpto)

Here are the detailed validation details about the three models.

**1.Encoder**

![](/images/enc.png)


**2.Encoder + LSTM**

![](/images/encoder+lstm.png)

**3.Encoder + Conv2D**

![](/images/encoder+cnn.png)


**Here are their areas under ROC curves**

| Model           	| ROC_AUC 	|
|-----------------	|---------	|
| Encoder         	| 0.906   	|
| Encoder + LSTM  	| 0.902   	|
| Encoder + Conv2D   	| 0.902   	|


# Model Selection [`↩`](#jumpto)

Although on first glance all three models have comparable metrics on further testing I found that the pure Encoder generalises the best. Again it's always best to choose a model based on an objective you want to maximize in the confusion matrix. In my case I was ready to compromise a bit on predicting sarcastic sentences if that meant I could decrease false positives. **The Encoder+LSTM model predicted sarcastic sentences very well but had high false positives. I created many sentences both sarcastic and otherwise and found that the pure Encoder generalises better than the other two**. 
I took alot of sarcastic sentences from this website.

![](https://parade.com/1079501/stephanieosmanski/sarcastic-quotes/)

**Here is the ROC curve of the final model in deployment.**

![](/images/roc.png)

# Results [`↩`](#jumpto)

Here are some results from the web app. 

The two sentences used in the images are

**Today is such a beautiful day , the weather is perfect to sit inside .**
![](/images/sarcq.png)

![](/images/sarca.png)



**Today is such a beautiful day , the weather is perfect for football .**

![](/images/noq.png)

![](/images/noa.png)


**Here are some more results**

| Sentence                                                       	| Prediction    	|
|----------------------------------------------------------------	|---------------	|
| I work forty hours a week for me to be this poor.              	| Sarcastic     	|
| I would kill for a Nobel Peace Prize.                          	| Sarcastic     	|
| Hello my name is Raj and I study computer science!             	| Not Sarcastic 	|
| It’s okay if you don’t like me. Not everyone has good taste.   	| Sarcastic     	|
| Well at least your mom thinks you’re pretty .                  	| Sarcastic     	|
| Marriage. Because your crappy day doesn’t have to end at work. 	| Sarcastic     	|
| I am so happy to be working with you guys.                     	| Not Sarcastic 	|
| Today was fun .                                                	| Not Sarcastic 	|
| It's nothing to joke about.                                    	| Not Sarcastic 	|
| Pleased to meet you.                                           	| Not Sarcastic 	|


# Usage and Tips [`↩`](#jumpto)

I would like to reiterate that the model is far from perfect. For best performance I would suggest you use examples which do not have any conversational dependence or history. They should summarize emotions in a single sentence (2 at maximum ). Please try to give as much context as possible since phrases such as **Sure!** , **What a day!** could be interpreted as both sarcastic and otherwise depending on the context. 

I will try to improve the model by training it on more examples in the future. The goal of this repo is only to implement and understand Encoders. For **practical usage using a pre-trained model such as BERT obviously yields better performance**

# Retrospection [`↩`](#jumpto)

While making this project, my goal was to try and understand state of the art NLP models better especially through a more mathematical perspective. All the references linked below have helped me immensely. This model is far from perfect and I will definitely try enhancing it after collecting more data. **20000** training sentences are too less to build a robust model from scratch(without pre-trained weights) in my opinion.


# References [`↩`](#jumpto)

[1](https://arxiv.org/pdf/1706.03762.pdf) Vaswani, Ashish & Shazeer, Noam & Parmar, Niki & Uszkoreit, Jakob & Jones, Llion & Gomez, Aidan & Kaiser, Lukasz & Polosukhin, Illia, “Attention is all you need” , 2017.

[2](https://arxiv.org/abs/1810.04805) Jacob Devlin,Ming-Wei Chang, Kenton Lee, Kristina Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" ,**v2 2019** 

[3](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270) Rani Horev, “BERT Explained: State of the art language model for NLP” blog post, 2018.

[4](http://jalammar.github.io/illustrated-transformer/) Jay Alammar, “The Ilustrated Transformer” blog post, 2018.

[5](http://peterbloem.nl/blog/transformers) Peter Bloem, “Transformers from scratch” blog post, 2019.

[6](https://towardsdatascience.com/attention-is-all-you-need-discovering-the-transformer-paper-73e5ff5e0634) Eduardo Munoz, “Attention is all you need: Discovering the Transformer paper” blog post, 2020. **Big Thank you!**

[7](https://stackoverflow.com/questions/58123393/how-to-use-transformers-for-text-classification) Jindrich's stackoverflow answer , 2019.






