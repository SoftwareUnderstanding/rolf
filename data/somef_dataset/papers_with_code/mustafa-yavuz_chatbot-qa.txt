# End-to-end memory network implementation for Question answering
This is an implementation of a chatbot that can answer questions based on a "story" given to the bot.

### Dataset
* [Babi](https://research.fb.com/downloads/babi/) dataset released by facebook research. 
* A particular subset of the dataset which has stories, questions and answers is used as data (One of 20 tasks in the bAbI project).
* Training set(10000) and test set(1000) are seperated and each sample is in a tuple format (story,question,answer)

### Model
<img src='https://i.imgur.com/0YVe2dY.png' title='Poster' width='' />

**a)** Single Layer case which implements a single memory hop operation

**b)** Multiple Layer implementation (using RNNs) with multiple hops in memory

**3 main sub-components of network**:
* Input Memory Representation 
* Output Memory Representation
* Generating Final Prediction
 
**Full model** : Using LSTMs with multiple layers on top of sub-components. Network produces a probabilty for every single word in the vocabulary. In this implementation, there will be high probablity on either yes or no.

Code accompanying the End-To-End Memory Networks paper: https://arxiv.org/pdf/1503.08895.pdf

For further understanding : https://www.youtube.com/watch?v=ZwvWY9Yy76Q


