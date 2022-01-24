# fake_news

This project focusses on applying a combination of BERT transformers from Hugging face, the Fast AI API library and various interpretability methods, including applying integrated gradients through Captum, to the Kaggle fake news dataset.

Model interpretability is growing in importance in a world increasingly governed by automated decisions. It is of particular value in a field such as 'fake news', where we are applying a boolean attribute to very complex attributes such as truth and reliability.

Dataset found here:
https://www.kaggle.com/c/fake-news/data

This repository demonstrates:

1. Training models using the BERT transformer and pytorch on labelled reliable/unreliable news sources.

2. Combining the Fast AI library and Transformers from the hugging face library to achieve high performance (would have placed 1st in Kaggle competition) and benefit from the ease of use of Fast AI, particularly in model training.

3. Model interpretability using hooks and the Captum library, based on "Axiomatic Attribution for Deep Networks".

Contents:

**Bert_Classifier:**
Simple BERT pytorch classifier trained on Kaggle dataset.

**FastAI+BERT**
Bert Transformer applied to a FastAI learner object. Model fine tuned to performance of >99% accuracy.

**Captum_interp**
Application of the Captum library and layered integrated gradients for model interpretability.


*Further Interpretablity work:*

In future I am very interested in Introspective Rationale and '3 player models' for text interpretability. This requires a generator model which selects text snippets to train two classifier models on. Aiming to optimise the accuracy of the model which is fed the selected text and minimise the accuracy of models fed deselected text. 

I have included my starter notebook on this topic, based on the github repo and research paper below.

References:

BERT - https://arxiv.org/abs/1706.03762

Integrated Gradients - https://arxiv.org/abs/1703.01365

Captum - https://captum.ai/tutorials/

FastAI + transformers - https://towardsdatascience.com/fastai-with-transformers-bert-roberta-xlnet-xlm-distilbert-4f41ee18ecb2

Introspective Rationale (paper) - http://people.csail.mit.edu/tommi/papers/YCZJ_EMNLP2019.pdf

Introspective Ratinale (interpret-text github) - https://github.com/interpretml/interpret-text

Good fake news reading - Cyberwar (Kathleen Jamieson)
