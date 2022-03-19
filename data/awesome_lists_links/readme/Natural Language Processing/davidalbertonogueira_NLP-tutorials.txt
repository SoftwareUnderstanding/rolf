# NLP-tutorials
Slides and code for the intro to NLP class at the Data Science Retreat  @ Berlin, 28 Oct 2019

## Installation

Clone this repo with 
```
git clone --recursive https://github.com/davidalbertonogueira/NLP-tutorials
git submodule update --depth 10 --recursive --init
```


### Requirements
Besides python >= 3.7 and pip, the required libraries are listed in `requirements.txt`.

Code is cross-platform. Tested in Windows and Ubuntu 18.04.

### Setting up a virtual environment
Linux  | Windows
------------- | -------------
sudo apt-get install virtualenv / pip install virtualenv | pip install virtualenv
virtualenv --python /usr/bin/python3.7 venv	  | virtualenv venv
source venv/bin/activate  | venv\Scripts\activate.bat
pip install -r requirements.txt  | pip install -r requirements.txt 
pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp37-cp37m-linux_x86_64.whl (1)* | pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp37-cp37m-win_amd64.whl (1)*
pip install torchvision  | pip install torchvision
pip install torchtext==0.2.3  | pip install torchtext==0.2.3 
python -m spacy download en_core_web_sm | python -m spacy download en_core_web_sm

(1)* replace "cpu" in link if you plan to use GPU: "cu80" for CUDA 8, "cu90" for CUDA 9.0, "cu92" for CUDA 9.2, ...

_If you require a newer version,
please visit http://pytorch.org/ and follow their instructions to install the relevant pytorch binary._


## Text Classification 
[Text Classification with BoW + TF-IDF + Naive bayes and SVM](code/TextClassification.ipynb)
based on http://github.com/Gunjitbedi/Text-Classification

[Text Classification with word embeddings + neural network](code/TextClassificationNN.ipynb)

### Sentiment Analysis
[Sentiment Analysis with task-specific word embeddings + Pytorch neural network](https://github.com/davidalbertonogueira/SentimentAnalysis)

The implemented model is a Deep Bidirectional LSTM model with Attention, based on the work of Baziotis et al., 2017: 
[DataStories at SemEval-2017 Task 4: Deep LSTM with Attention for Message-level and Topic-based Sentiment Analysis](http://aclweb.org/anthology/S17-2126).

## Recomendation 
[Recomendation using Colaborative filtering](code/Recomendation.ipynb)

## Named Entity Recognition 
[Transition-based NER system (in C++ Dynet)](https://github.com/davidalbertonogueira/stack-lstm-ner-dynet)
[(Pytorch version)](https://github.com/davidalbertonogueira/stack-lstm-ner-pytorch)

Paper: https://arxiv.org/pdf/1603.01360.pdf

## Bonus material
[Transformers: Attention is all you need](https://github.com/davidalbertonogueira/annotated-transformer/blob/master/The%20Annotated%20Transformer.ipynb)
copied from http://nlp.seas.harvard.edu/2018/04/03/attention.html 

Paper: https://arxiv.org/abs/1706.03762
