## Introduction

Chat-bots are becoming more and more useful in various simple professional tasks as they get more and more able to capture the essence of communicating with people. Still the development of good chat-bots that will answer more complicated questions, in general subjects, is a growing domain of research. 

The goal of this project is to create a chat-bot able to answer python related questions.  Our project started with the main idea being that a programming assistant would be a much needed help by many people working or studying computer science. Although it sounds simple it soon proved to be a difficult task. The main challenge is that the model has to extract a technical correlation between Questions and Answers in order to be able to communicate effectively. The model that we used in order to achieve our goal is a recurrent sequence-to-sequence model. The main steps that we followed are described bellow.

- We found, downloaded, and processed data taken from stack overflow concerning questions that contained at least one python tag.[7]
- Implement a sequence-to-sequence model.
- Jointly train encoder and decoder models using mini-batches
- Used greedy-search decoding
- Interact with trained chatbot

## Table of Contents

1. [Execution](#execution)
2. [Pre-Processing](#pre-processing)
3. [Data Preparation](#datapreparation)
4. [Models](#models)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Results](#results)
8. [References](#references)

## <a name="execution"></a> 1. Execution

For this project we used the pytorch framework of python. The project code is  located in the Python-Chatbot.ipynb  jupyter notebook. It was executed in Google Colab environment, using a GPU. 

Anyone who wants to run the code can do it from the beginning or if a pre-trained model is available they can jump directly to the part where the model is loaded. Comments inside the notebook explain which parts can be skipped.

Since data files are very large (approximately 800Mb each) we are not going to upload them on this repository. Instead we provide the download link in the References section and we suggest that they should be uploaded to a goole drive account. Then the google account can be easily connected with the colab platform in order for the files to be loaded. Code for this purpose already exists in the jupyter notebook. 

## <a name="pre-processing"></a> 2. Pre-Processing

Data preprocessing is done in two phases.

- Phase 1 Read row data and:
  - Keep all questions with at least one answer.
  - Pair each question with its most up voted answer.
  - Remove all questions that need code in order to be answered.

That last step is needed in order to simplify the task, as feeding code blocks to the model would require special handling.

- Phase 2:
  - Remove punctuation and special characters.
  - Remove HTML and Markdown tags.
  - Filter sentences with length greater than a given value.
  - Filter pairs containing  rare words (words with an appearance frequency lower than a given value).

## <a name="datapreparation"></a> 3. Data Preparation

Now it is time to prepare our data to be fed in to the model. For this reason the following steps are followed:

- Create torch tensors of data.
- Create tensors of shape (max_length, batch_size) in order to help train using mini-batches instead of 1 sentence at a time. 

- Zero pad tensors to fit the maximum sentence length.
- Create tensors of length for each sentence in the batch.
- Create mask tensors with a value of  1 if token is not a PAD_Token else value is 0.

 

## <a name="models"></a> 4. Models

We use a sequence two sequence (seq2seq) model composed from 2 Recursive Neural Networks (RNNs) one acting as an encoder and the other acting as a decoder.

- Encoder:

  The Encoder iterates through the input sentence one word at a time, at each time step outputting an:

  -  Output vector. 
  - Hidden state vector. 
  - We used a bidirectional variant of the multi-layered Gated Recurrent Unit [4], 
  - Two independent RNNs.
  - One that is fed the input sequence in normal sequential order. 
  - And one that is fed the input sequence in reverse order. 
  - The outputs of each network are summed at each time step. Using a bidirectional GRU.

- Decoder:

  The decoder RNN generates the response sentence in a token-by-token fashion using:

  - Context vectors
  - Internal hidden states

    from the encoder to generate the next word in the sequence. 

  In order to minimize information loss during encoding process we will use the **Global attention** mechanism by [5] which improved upon Bahdanau et al.’s [6] attention mechanism.



So the flow of our seq2seq model is:

1. Get embedding of current input word. 
2. Forward through unidirectional GRU. 
3. Calculate attention weights from the current GRU output.
4. Multiply attention weights to encoder outputs to get new "weighted sum" context vector. 
5. Concatenate weighted context vector and GRU output using Luong. 
6. Predict next word. 
7. Return output and final hidden state.

## <a name="training"></a> 5. Training

The training procedure consists of the following steps:

1. Forward pass entire input batch through encoder. 

2. Initialize decoder inputs as SOS_token, and hidden state as the encoder's final hidden state. 

3. Forward input batch sequence through decoder one time step at a time. 

4. If teacher forcing: set next decoder input as the current target else set next decoder input as current decoder output. 

5. Calculate and accumulate loss. 

6. Perform backpropagation. 

7. Clip gradients. 

8. Update encoder and decoder model parameters.

   

During the training process we use a these tricks to aid in convergence:

- **Teacher forcing:** At some probability, set by **teacher_forcing_ratio**, we use the current target word as the decoder’s next input rather than using the decoder’s current guess. 
- **Gradient clipping**. Commonly technique for countering the “exploding gradient” problem. In essence, by clipping or thresholding gradients to a maximum value, we prevent the gradients from growing exponentially and either overflow (NaN), or overshoot steep cliffs in the cost function.

## <a name="evaluation"></a> 6. Evaluation

Evaluation Decoding Flow:

**Decoding Method**

1. Forward input through encoder model. 
2. Prepare encoder's final hidden layer to be first hidden input to the decoder.
3. Initialize decoder's first input as SOS_token. 
4. Initialize tensors to append decoded words to. 
5. Iteratively decode one word token at a time: 
   1. Forward pass through decoder. 
   2. Obtain most likely word token and its softmax score. 
   3. Record token and score. 
   4. Prepare current token to be next decoder input. 

6. Return collections of word tokens and scores.



Greedy decoding

- Greedy decoding is the decoding method that we use during training when we are **NOT** using teacher forcing. 
- For each time step we choose the word from decoder_output with the highest softmax value. 
- This decoding method is optimal on a single time-step level.



Evaluation Process

- Format the sentence to be evaluated as an input batch of word indexes with *batch_size==1*. 
- Create a **lengths** tensor which contains the length of the input sentence.
- Obtain the decoded response sentence tensor using **GreedySearchDecoder**. 
- Convert the response’s indexes to words and return the list of decoded words.
- When chatting with the bot this evaluation process is followed in order for it to respond.



## <a name="results"></a> 7. Results

Experiment results confirm the this is a complicated task and that further work may still to be done. Bellow are some good and some bad examples from different training and executions of the program:

- Good results

  <img src="./images/good_res1.png" alt="alt text" width="400" height="100" />

  <img src="./images/good_res2.png" alt="alt text" width="400" height="100" />

- Bad results

  <img src="./images/bad_res1.png" alt="alt text" width="300" height="100" />

  <img src="./images/bad_res3.png" alt="alt text" width="400" height="100" />

## <a name="references"></a> 8. References

1. ChatbotTutorial by Matthew Inkawhich  
   https://pytorch.org/tutorials/beginner/chatbot_tutorial.html​
2. Pytorch Chatbot by Wu,Yuan-Kuei 
   https://github.com/ywk991112/pytorch-chatbot​
3. Sutskever et al. 
   https://arxiv.org/abs/1409.3215​
4. Cho et al. 
   https://arxiv.org/pdf/1406.1078v3.pdf​
5. Luong et al. 
   https://arxiv.org/abs/1508.04025​
6. Bahdanau et al 
   https://arxiv.org/abs/1409.0473​
7. Python Questions from Stack Overflow 
   https://www.kaggle.com/stackoverflow/pythonquestions​

