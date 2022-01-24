# Chat-Bot

The Chat Bot is build on a part of the 
### bAbI Dataset released by Facebook Research
link - https://research.fb.com/downloads/babi/

## ------------------------------------------------------------------

#### To Build the Chat Bot I followed the Research Paper
### End-To-End Memory Networks by
### Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston & Rob Fergus

link1 - http://arxiv.org/abs/1502.05698

link2 - http://arxiv.org/abs/1503.08895


## ------------------------------------------------------------------

A single layer model takes a discrete set of inputs x1; :::; xn that are to be stored in the memory, 
a question q, and outputs an answer a. Each of the xi, q, and a contains symbols coming from a dictionary with V
words(vocab). The model writes all x to the memory up to a fixed buffer size, and then finds a continuous
representation for the x and q. The continuous representation is then processed via multiple hops to
output a(LSTM Layers are used). This allows backpropagation of the error signal through multiple memory accesses back
to the input during training.

#### The Research Paper must be read to get a understanding on how the Model Works
#### And why we are using this Set Up
