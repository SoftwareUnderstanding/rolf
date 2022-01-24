# COVID19-document-retrieval-with-BERT
This project is about developing a document retrieval system to return titles and the context of scientific papers containing the answer to a given user question.
We will be using CORD-19 [CORD-19 dataset](https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases.html) for the dataset which is the biggest corpus of academic papers about COVID-19 and related coronavirus research.

## About the project

In this project, we will use and evaluate 4 different methods for creating contextual sentence embeddings. These embeddings then will be used to find the most semantically meaningful document that relates to our question. 

The 4 methods are:

*  pre-trained BERT average output layer as embeddings
*  GloVe embeddings
*  pre-trained BERT large model on a corpus of messages from Twitter about COVID-19
*  pre-trained Sentence-BERT embeddings


## About the implementation

Firstly the data is extracted from the dataset and a Pytorch dataset class is used to access the corpus. Then each model is defined separately. The corpus then is batched in sentences and batches of sentences are
feed in every model. From the output of the models' embeddings are generated for every sentence in the documents and they are stored and zipped in the cloud.
After the transformation of the corpus into vectors of words, follows the question answering part. 
We download and unzip the embeddings of all the documents per each embedding model method. Afterward, we ask for a question. We transform the question into an 
embedding vector and we search with cosine similarity through the dataset the most similar sentence embedding to our question for each model method.
Finally, we can evaluate our results.


## Code and report
Code, comments and report exist in the notebook which you can open in Colab from here [![Click here to open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AGiannoutsos/COVID19-document-retrieval-with-BERT/blob/main/covid_document_retrieval.ipynb)



## Resources

CORD-19 dataset: https://www.aclweb.org/anthology/2020.nlpcovid19-acl.1.pdf

S-BERT paper: https://arxiv.org/pdf/1908.100Bert84.pdf

GloVe: https://nlp.stanford.edu/projects/glove/

Ηuggingface library: https://huggingface.co/bert-base-uncased

ΒΕΡΤ model paper: https://arxiv.org/abs/1810.04805

BERT pretrained on Twiiter for COVID related tweets paper: https://github.com/digitalepidemiologylab/covid-twitter-bert
