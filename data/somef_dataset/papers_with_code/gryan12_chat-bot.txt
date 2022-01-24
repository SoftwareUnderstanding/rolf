# qna chat-bot
simple question and answer chat bot using keras, tensorflow backend

The aim was to make a simple commandline interface using keras with tensorflow backend where the user could ask simple questions that the bot would answer. 

An end-to-end neural network was used, as outlined in the following paper: https://arxiv.org/pdf/1503.08895.pdf
The Babi data set from facebook research was used for training https://research.fb.com/downloads/babi/

The program takes in questions built from its vocab list, and returns (if all works) either yes or no. 

Due to known issues with saving and loading keras model weights between sessions, currently the bot interface is coded in the same file as 
the one that trains it. Once I have found a fix/workaround for loading model weights, I will upload a number of different trained model weights and allow
the user to select which one that their qna bot will use without them having to sit through model training. 

save_weights and load_model issues: 
  https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras/issues/13
  https://github.com/keras-team/keras/issues/3927
  https://github.com/keras-team/keras/issues/4904
