# Ramdomized-Clinical-Trail-Classification

This is the RCT classification part of our Capstone Project at Harvard 599 course. I am the major contributor of this part, from literature review, data preparation, designing models, coding, training model, to tuning models.

## Overview

Randomized clinical trials (RCTs) are imperative to the progress of medical research, with thousands being published on an annual basis. Currently, the number of published RCTs is over one million, and half of those RCTs are found in PubMed, a freely accessible search engine primarily accessing the MEDLINE database of biomedical literature. While these RCTs are freely available to the public, there is currently no straightforward way to identify RCTs amongst the multitude of article types in the database. Identifying RCTs efficiently presents itself as an ongoing challenge for medical researchers.

In this project, various deep learning models are tested to identify and classify RCTs. Unlike other related work which use the abstract portions of the papers only, our work is based on experiments using both the full text documents and their abstracts. The selected deep learning model combines a long short-term memory (LSTM ) model and one-dimension convolution together to process the vectors generated from Doc2Vec. This model can classify articles with a relatively high accuracy and low computational requirement. 

The majority deelp learning models are developed on MXNET, Keras, and Tensorflow. Transfer learning is also used to compare results.

## Why this project is different from previous works.

Previous work on RCT classification were based on the abstracts [Marshal et al. 2017] of RCT and other medical documents. This approach has two limitations. First, not all the articles have abstract in the PMC database. Second, medical researchers usually are more interested to have a tool to separate RCTs from other clinical trials. This work is more challenging. 

Below figure illustrates the difference between our work and other works. It is a two-dimension projection of the TFIDF of 300 articles randomly selected from PMC, which contain 100 RCT, 100 non-randomized clinical trials, and 100 other medical documents. The yellow circles represent RCTs; the blue rectangles represent nonRCTs; the red triangles represent other medical documents. Even an unsupervised machine learning model can separate the RCTs from other documents well; however, to separate the highly overlapping RCTs and nonRCTs is more challenging.

![alt text](https://github.com/DCYN/Ramdomized-Clinical-Trail-Classification/blob/master/Figure1.png)

## Methodology

The entire process of developing an RCT classification tool can be separated into four distinct steps: prepare the dataset and extract data, tokenize of sentences, vectorize of tokens, and finally classification. These parts are illustrated as below:

![alt text](https://github.com/DCYN/Ramdomized-Clinical-Trail-Classification/blob/master/Process.png)

TFIDF, Word2Vec, and Doc2Vec are tested for vectorization. 

SVM, SVM+DNN, SVM+CNN, 1dCNN, 1dCNN+ LSTM are tested as classifiers. Below are a few samples of the classifiers.

![alt text](https://github.com/DCYN/Ramdomized-Clinical-Trail-Classification/blob/master/Classifiers.png)

## Result summary
### 1.	Abstract only classification.

TFIDF-Naïve Bayes outperform the other models on abstracts only data.

Naïve Bayes model performs better on abstracts only data. It yields a F1-score of 81%. This combination also yields the best results when only using the abstract portions of the articles. The results validate the findings of Wang [Wang, Sida, 2012] that Naive Bayes can yield better results on snippets.

The vectorization for all the other classifiers are Doc2Vec with following hyper-parameters: size = 1024, Window = 3, Iter = 30, Min Count =2. 

![alt text](https://github.com/DCYN/Ramdomized-Clinical-Trail-Classification/blob/master/Abstracts_result.png)

### 2. Full text classification with unbalanced sampling. (SMOTE)

Doc2Vec + Inception batch normalization outperforms the other models for the dataset excluding the nonRCT provided by PubMiner. SMOTE is used to do unbalanced sampling.

Doc2Vec can generate the vectors to fed the SVM and the deep learning models we tested. If we only use the labeled data gotten from clinicaltrails.gov,  the whole dataset contains 10,216 full text articles, 1,627 Non-RCTs and 8,589 RCTs. Since the dataset is highly unbalanced, the models other than the LSTM tends to give all positive prediction if unbalanced sampling is not used. 

We can use SMOTE to solve the all positive prediction issue, but the results indicate overfitting. 
![alt text](https://github.com/DCYN/Ramdomized-Clinical-Trail-Classification/blob/master/wo_pubminer.png)

### 3.	Create balanced data set.

Doc2Vec + 1dCNN + LSTM outperforms the other models for the dataset including the nonRCT provided by PubMiner. 

Although training a model using unbalanced sampling results in higher F1 score, further verification using real world data is desirable. After incorporating the nonRCTs provided by PubMiner, we get below results.

![alt text](https://github.com/DCYN/Ramdomized-Clinical-Trail-Classification/blob/master/w_pubminer.png)

### 4. Using Word2Vec.

In order to compare our research with others’ work. We apply the CNN model [Marshal et al. 2017] on our data set. The pre-trained vectors generated from Word2Vec For Word2Vec, a pre-trained RCT sentence vectors [Dernoncourt, Franck, 2017]  is used, and the sequence is set to 1000. This model did not outperform the model using Doc2Vec vectors in the experiments. One potential explanation may be that this vector was trained using sentences from the abstract portions of the articles only. Another possible reason is that semantics is important for a classifier to evaluate the possibility of a document to be RCT, while Doc2Vec processes semantics better than Word2Vec [Mikolov, Tomas 2014].  

The 1-D convolutional layer is important with the long word sequences used. This was in order to reduce the dimensions of the vectors to use in an LSTM model and therefore to maintain performance while reducing both training time and resource needs.

We also feed a 1dCNN model and the 1dCNN + LSTM model with the pre-trained vectors generated by word2vec. Both model outperform the CNN model by Ian Marshall.

![alt text](https://github.com/DCYN/Ramdomized-Clinical-Trail-Classification/blob/master/word2vec.png)

### 5. Doc2Vec vs. Word2Vec.

Smaller vector size with better optimized hyper-parameters for Doc2Vec.

Although Doc2Vec can help to get better performance, it’s still time consuming to generate the vectors with big size. During the experiments, we find that, instead of increasing the vector size, adjusting other hyperparameters such as window, min_count, and epoch can yield good results with fewer time a memory resource consumption.

A vector size of 256 can generate the favorable results as long as other parameters are set appropriately. Setting a min_window of 1 can help the Doc2Vec vectors capture information more sensitively while increasing the window of context length to 21 helps to preserve more syntax information. The epochs were set to 50 in the final solution.

![alt text](https://github.com/DCYN/Ramdomized-Clinical-Trail-Classification/blob/master/doc2vsword2.png)

Detailed configuration of models can be found in the subfolders.

Future Work
Due to a limited amount of time, our experiments do not cover the full range of combinations of both the deep network architectures and the methods to vectorize the text content. My experiment after this course revealed that the LSTM model can be simplified further to get even better results.

A new development of language process comes from deep learning sequential models with an attention component [Dzmitry Bahdanau, 2014]. Attention models were not investigated.

## References:

1.	Iain J. Marshall, Anna Noel-Storr, Joël Kuiper, James Thomas,  Byron C. Wallace, (2017) Machine learning for identifying Randomized Controlled Trials: An evaluation and practitioner's guide. https://onlinelibrary.wiley.com/doi/full/10.1002/jrsm.1287

2.	Franck Dernoncourt, Frank, Lee Ji Young (2017),  PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts. Retrieved from https://arxiv.org/pdf/1710.06071.pdf

3.	Efsun Sarioglu, Kabir Yadav, Topic Modeling Based Classification of Clinical Reports, http://www.aclweb.org/anthology/P13-3010

4.	Ting, S. L., Ip, W. H., Tsang, A. H. (2011). Is Naive Bayes a good classifier for document classification, https://www.researchgate.net/publication/266463703_Is_Naive_Bayes_a_Good_Classifier_for_Document_Classification

5.	Sida Wang and Christopher D. Manning, Baselines and Bigrams: Simple, Good Sentiment and Topic Classification,  https://www.aclweb.org/anthology/P12-2018

6.	Mengen Chen, Xiaoming Jin, Short Text Classification Improved by Learning Multi-Granularity Topics, http://www.ijcai.org/Proceedings/11/Papers/298.pdf

7.	Kunlun Li, Jing Xie, Multi-class text categorization based on LDA and SVM, https://www.sciencedirect.com/science/article/pii/S1877705811018674

8.	Qiuxing Chen, Lixiu Yao, 2016, Short text classification based on LDA topic model, https://ieeexplore.ieee.org/document/7846525/

9.	Eugene Nho, Andrew Ng., Paragraph Topic Classification 

10.	http://cs229.stanford.edu/proj2016/report/NhoNg-ParagraphTopicClassification-report.pdf

11.	Andrew Ng. Sequence Models, https://www.coursera.org/learn/nlp-sequence-models

12.	 Spärck Jones, K. (1972). "A Statistical Interpretation of Term Specificity and Its Application in Retrieval". Journal of Documentation. 28: 11–21. doi:10.1108/eb026526.

13.	Luhn, Hans Peter (1957). "A Statistical Approach to Mechanized Encoding and Searching of Literary Information" (PDF). IBM Journal of research and development. IBM. 1 (4): 315. doi:10.1147/rd.14.0309.

14.	Mikolov, Tomas; et al. "Efficient Estimation of Word Representations in Vector Space". https://arxiv.org/abs/1301.3781

15.	Quoc Le and Tomas Mikolov. Distributed Representations of Sentences and Documents. http://arxiv.org/pdf/1405.4053v2.pdf

16.	Sepp Hochreiter; Jürgen Schmidhuber (1997). "Long short-term memory". Neural Computation. 9 (8): 1735–1780. doi:10.1162/neco.1997.9.8.1735. PMID 9377276.

17.	Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio, Neural Machine Translation by Jointly Learning to Align and Translate, https://arxiv.org/abs/1409.0473
