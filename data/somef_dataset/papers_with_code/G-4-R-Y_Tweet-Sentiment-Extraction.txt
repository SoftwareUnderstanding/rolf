## Tweet-Sentiment-Extraction
*Text extraction given sentiment using roberta model*

For this experiment, it was used 5 fold cross validation in order to train a roberta model that, given a tweet and sentiment, can extract text that justifies the given sentiment.

The overall Jaccard score on the task was 0.6987

In order to learn the fundamentals of attention and BERT based models, I strongly recommend reading Jay Alammar's blog posts http://jalammar.github.io/

Also, there are always the original papers: 
-Attention is All You Need: https://arxiv.org/abs/1706.03762
-RoBERTa: A Robustly Optimized BERT Pretraining Approach: https://arxiv.org/abs/1907.11692

The original kaggle kernel can be accessed through the link https://www.kaggle.com/ricafernandes/tweet-text-extraction-given-sentiment-roberta

*Future plans*: 
* fine tune the model
* implement roberta using fast autoregressive transformers with linear attention. By doing so, a huge performance boost is expected, paired with a slight loss of performance, hopefully turning roberta viable in real-time applications.
