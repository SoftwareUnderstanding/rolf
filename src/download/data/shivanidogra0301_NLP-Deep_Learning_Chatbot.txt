# Deep Learning Question & Answer Bot Project
# Project Overview
In this project, we implement a chatbot that can answer questions based on a "story" given to the bot.

We are using a subset of the BaBi dataset released by Facebook research. https://research.fb.com/downloads/babi. There are 10,000 data in the training set and 1,000 data in the testing set. Each data in the training/testing set consists of 3 components:

- Story - consists of single or multiple sentences
- Question - single sentence query related to the story
- Answer - "yes" or "no" answer to the question


The model for our chatbot is a RNN network with attention mechanism. It includes the following layers: Embedding, LSTM, Dropout, Dense and Activation. The design of the model pretty much follows the idea in the paper "End-to-End Memory Networks": https://arxiv.org/pdf/1503.08895.pdf.

Our model achieved pretty high accuary on training/testing set and performs really good on run time generated data.
