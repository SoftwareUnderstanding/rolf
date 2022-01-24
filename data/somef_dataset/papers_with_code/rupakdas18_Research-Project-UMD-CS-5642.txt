# Research-Project-UMD-CS-5642


This project describes the BERT model which is a transformer-based architecture.  BERT has been used in task 4(A,B,C), English Language, Sentiment Analysis in Twitter of SemEval2017. BERT is a very powerful model for classification tasks when the number of training data is small. For this experiment, we have used BERTBASE model which has 12 hidden layers. This model provides better accuracy, precision, recall, f1 score than the Naive Bayes baseline model. It performs better in binary classification subtask than the multi-class classification subtasks. We also considered all kinds of ethical issues during this experiment as Twitter data contains personal and sensible information.

**Task Description:**

1. **Subtask A:**
Given a message, classify whether the message is of positive, negative, or neutral sentiment.
2. **Subtask B:** 
Given a message and a topic classify based on a two-point scale (positive or negative).
3. **Subtask C:** 
Given a message and a topic classify based on a five-point scale.


**Proposed Methodology:**

We are going to use the BERT model with the hugging face framework which is available in https://huggingface.co/transformers/model_doc/bert.html. BERT model is actually a multi-layer bidirectional Transformer encoder. The Transformer architecture is described in https://arxiv.org/pdf/1706.03762.pdf. It is an encoder-decoder network that uses self-attention on the encoder side and attention on the decoder side. The Transformer reads entire sequences of tokens at once while LSTMs read sequentially. BERT has 2 model sizes. One is BERT-BASE that contains 12 layers and another one is BERT-LARGE has 24 layers in the encoder stack. In the Transformer, the number of layers in an encoder is 6. After the encoder layer, both BERT model has Feedforward-network with 768 and 1024 hidden layer, respectively. Those two models have more self-attention heads (12 and 16 respectively) than Transformer. BERT-BASE contains 110M parameters while BERT-LARGE contains 340M parameters.

This model has 30,000 token vocabularies. It takes ([CLS]) token as input first, then it is followed by a sequence of words as input. Here ([CLS]) is a classification token. It then passes the input to the above layers. Each layer applies self-attention, passes the result through a feedforward network after then it hands off to the next encoder. The model outputs a vector of hidden size (768 for BERT-BASE and 1024 for BERT-LARGE ). If we want to output a classifier from this model, we can take the output corresponding to [CLS] token.

BERT is pre-trained using Masked Language Modeling (MLM) and Next Sentence Prediction (NSP) on a large corpus from Wikipedia.  BERT was trained by masking 15% of the tokens with the goal to guess those words. For the pre-training corpus authors used the BooksCorpus which contains 800M words and English Wikipedia which contains 2,500M words.

BERT is helpful for different Natural Language Processing tasks like classification tasks, Name Entity Recognition, Part of Speech tagging, Question Answering, etc. But it is not useful for Language Models, Language Translation or Text Generation. BERT model is large and takes time for fine-tuning and inferencing.


**Baseline Method:**

After implementing a model, we want to know whether it performs better than the other models especially if there is a simpler or more tractable approach. This approach is referred to as the baseline. For a baseline proof-of-concept model, we will use a Naive Bayes Classifier which is often a good choice as a baseline model. This classifier is a simple classifier for the classification based on probabilities of a particular event. It is normally used in text classification problems. It takes less training time and less training data, less CPU and memory consumption.

For the evaluation measures for the three subtasks, we have used accuracy, precision, recall, and F1 score.
