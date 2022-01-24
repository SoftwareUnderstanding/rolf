# Assignment-2-Neural-Machine-Translation
Assignment 2: Neural Machine Translation in 'Statistical Machine Translation' course by Roee Aharoni at Bar-Ilan University.

link to the instructions file: https://docs.google.com/document/d/1f1J_eBfLXMaEx5r52xV-LfrnTKJ-qmui7exfA0LNHIg/edit?usp=sharing

In this assignment, I had to implement an Neural machine translation (NMT) system using a deep learning framework - PyTorch.

In the 1st part of this exercise, I had to implement an RNN based encoder-decoder model for the task of decoding given number sequences into character sequences.
The “connection” between the encoder and the decoder, in this part, was by using the last encoder output as an input to each decoder state - by concatenating it to the embedding of the previous output symbol.

In the 2nd part, I had to implement an encoder-decoder model with an attention mechanism. The attention mechanism allows the decoder to learn a different representation of the source sentence in every decoder step, by computing attention weights for each input element. Once the attention weights are computed, they are used to compute a context-vector which is a weighted average of the encoder representations. This context vector is then used to compute the decoder output for this step.
The model I had to build in this part was “Global-Attention” model, according to the terminology of the paper https://arxiv.org/pdf/1508.04025.pdf by Loung, Pham and Manning (2015).

In the 3rd part, I had to analyze the attention weights of the model from Part 2. I added code to the training script, that dumps the attention weights for a specific example from the development set after each epoch. And then, visualizes the attention weights for the example after each epoch using a heatmap plot.

Score: 100
