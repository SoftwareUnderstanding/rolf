# Review 3 Different Methods for Explainability Artificial Intelligence (XAI)

Some plot use javascript since github lock javascript for protection here link to Google Colab for better visualization

[USAIR](https://colab.research.google.com/drive/1wRCuvkmMa92okus_YA0CwT8-IyLcB0cK?usp=sharing)

[Indonesian review](https://colab.research.google.com/drive/1fgEyMDFOfJkZFA9Up1l8J5-qCef-F4dY?usp=sharing)

In this repository, we use 3 different methods of XAI on Machine Learning Model for Text Classification (Sentiment classification using Bi-LSTM and BI-GRU+LSTM+CNN). 

Explainable Methods we use :

* LIME explanation method for text:Local interpretable model-agnostic explanations (LIME) are a concrete implementation of local surrogate models. Surrogate models are trained to approximate the predictions of the underlying black box model. Instead of training a global surrogate model, LIME focuses on training local surrogate models to explain individual predictions. LIME for text has variations of the data which are generated differently: Starting from the original text, new texts are created by randomly removing words from the original text. The dataset is represented with binary features for each word. A feature is 1 if the corresponding word is included and 0 if it has been removed.
* SHAP (kernel Explanation) method for text: SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model.SHAP has the goal to explain the prediction of an instance x by computing the contribution of each feature to the prediction
* Anchor explanation method: The anchor algorithm is based on the Anchors: High-Precision Model-Agnostic Explanations paper by Ribeiro et al. For text classification, an interpretable anchor consists of the words that need to be present to ensure a prediction, regardless of the other words in the input.

To compare the methods we use the commons datasets :

* Review of Indonesian Application in google playstore for classification dataset 
* US Airlines for Sentiment classification [dataset](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)


Contributor:
* [Nadhila](https://github.com/Nadhila)
* [Dimdimadi](https://github.com/dimdimadi)


## Model description:

### Data Cleaning

* Lowering case :  We decide to reduce the noise by normalizing each word to be lowercase
*	Punctuation removing : We removed punctuation such as question marks, commas, colons and periods.
* url removing (tweet data) : All instances of real URLs were removed to reduce the noise of the data.
*	Mention removing  and hashtag (tweet Data) : Tweet content data consisted some metion to other user and hashtag. These user mentions and hashtag which would be of little value to the model, and thus all such mentions were removed.

### Preprocessing Data : Sequence Padding 


  Padding is a special form of masking were the masked steps are at the start or at the beginning of a sequence. Padding comes from the need to encode sequence data into contiguous batches: in order to make all sequences in a batch fit a given standard length, it is necessary to pad or truncate some sequences.
When processing sequence data, it is very common for individual samples to have different lengths.A requirement for our LSTM models was that the input sentence length has to be ﬁxed for all training dataset. We use sequence_pads () function from keras with maxlen= input_length which is 100. 


### Embedding (embedding layer) word2vec:

  We trained our own word embeddings during the execution of the whole model, with a randomly initialized embedding layer.

### Model Architecture:

We used two different deep recurrent network architectures, they are :

### Bidirectional LSTM 
The model are inspired by the paper Bidirectional Recurrent Models for Offensive Tweet Classiﬁcation

We build model which are contains layers:
-	1 Embedding layer responsible for the word embedding
-	1 spatialDropout1D layer Decreasing the number of features that we train on
-	Bidirectional LSTM layer A variant of the LSTM That uses two LSTMs one forward and one backward.
- Flatten layer
-	2 Dense layer.Last dense layer with softmax
-	1 dropout layer with 50% which located between the dense layer 

The details architecture of the model bi-directional (picture)

![Alt text](https://github.com/Nadhila/Explainble-AI/blob/master/bi-LSTM-model.png "Bi-LSTM Model")


### Model 2 bi gru +LSTM +cnn

We build model which are contains layers:
-	1 Embedding layer responsible for the word embedding
-	1 spatialDropout1D layer Decreasing the number of features that we train on
-	Bidirectional GRU layer A variant of the LSTM That uses two LSTMs one forward and one backward.
-	1D Convolutional layer
-	1 Global average polling 1D layer and 1 global max polling 1D layer  (this two layer are concatenate to be one layer) 
-	2 dense layer. Last dense layer with softmax
-	1 dropout layer with 50% which located between the dense layer


The details architecture of the model Bi-GRU+LSTM+CNN (picture)


![Alt text](https://github.com/Nadhila/Explainble-AI/blob/master/bi-GRU-LSTM-CNN.png "Bi-GRU+LSTM+CNN Model")


## References
Assawinjaipetch, P., Shirai, K., Sornlertlamvanich, V., & Marukata, S. (2016, December). Recurrent neural network with word embedding for complaint classification. In Proceedings of the Third International Workshop on Worldwide Language Service Infrastructure and Second Workshop on Open Infrastructures and Analysis Frameworks for Human Language Technologies (WLSI/OIAF4HLT2016) (pp. 36-43).(https://www.aclweb.org/anthology/W16-5205/)

Cambray, A., & Podsadowski, N. (2019). Bidirectional Recurrent Models for Offensive Tweet Classification. arXiv preprint arXiv:1903.08808. (https://arxiv.org/abs/1903.08808)
Ding, Z., Xia, R., Yu, J., Li, X., & Yang, J. (2018, August). Densely connected bidirectional lstm with applications to sentence classification. In CCF International Conference on Natural Language Processing and Chinese Computing (pp. 278-287). Springer, Cham. (https://link.springer.com/chapter/10.1007/978-3-319-99501-4_24)

Hu, J. (2018). Explainable Deep Learning for Natural Language Processing. (http://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1335846&dswid=1510)

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. In Advances in neural information processing systems (pp. 4765-4774). (http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predicti)

Mathews, S. M. (2019, July). Explainable Artificial Intelligence Applications in NLP, Biomedical, and Malware Classification: A Literature Review. In Intelligent Computing-Proceedings of the Computing Conference (pp. 1269-1292). Springer, Cham.(https://link.springer.com/chapter/10.1007/978-3-030-22868-2_90)

Molnar, C. (2020). Interpretable machine learning. Lulu. com.(https://christophm.github.io/interpretable-ml-book/)

Ribeiro, M. T., Singh, S., & Guestrin, C. (2018, April). Anchors: High-precision model-agnostic explanations. In Thirty-Second AAAI Conference on Artificial Intelligence.(https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPaper/16982)

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016, August). " Why should i trust you?" Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 1135-1144).(https://arxiv.org/abs/1602.04938)

Van Huynh, T., Nguyen, VD, Van Nguyen, K., Nguyen, NLT, & Nguyen, AGT (2019). Hate Speech Detection on Vietnamese Social Media Text using the Bi-GRU-LSTM-CNN Model. arXiv preprint arXiv: 1911.03644 . (https://arxiv.org/abs/1911.03644)

(https://github.com/slundberg/shap)
