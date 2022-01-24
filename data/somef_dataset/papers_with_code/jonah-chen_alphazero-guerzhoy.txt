# alphazero-guerzhoy

Mastering gomoku using the General Reinforcement Learning Algorithm from DeepMind in the paper published here: https://arxiv.org/pdf/1712.01815.pdf, https://doi.org/10.1038/nature24270. 

### 8x8 Gomoku
As this is an extention of a class project, the version with the 8x8 board is attempted first. The ResNet from the paper is built with a keras model and trained with the self play data using the same Monte-Carlo tree search algorithm. The board is represented of two 8x8 feature board that one-hot encodes the stones each player has already played.  

The only knowledge given to the agent is the condition for a win (5 or more stones in a row) and the basic rules (like one stone can be placed on each turn and you can't remove opponent's stones etc). From that, it was able to learn everything we humans do when playing the game like blocking the opponents "semi-open" sequence at 4 long, "open" sequence at 3 long as well as more advanced techniques like making "double-threats" to win. Although in 2020 this is like "normal" I still find it quite incredible that these behavior can be seen by just an optimization algorithm.  

The agent has trained over just under 100 batches of self play consisting of 200 games each, (under 20000 games). These training data can be seen from selfplay_data folder as .npy files with s (the state of the board), pie (the desired policy obtained from MCTS), and z (the value i.e. the outcome of the game). These games are played by the "best agent" which is updated if it is defeated by more than 55%. These games between different agents are also provided in the games folder and labelled ({black_model_no}v{white_model_no}.npy) and can be browsed using the analysis.py file to be displayed with a GUI interface.  

You can play the A.I. by executing the play_alphazero-guerzhoy.py file in a GUI enviroment. Only the latest model is uploaded in the models folder (as github LFS doesn't let me upload them all). <a href="https://mega.nz/folder/RpslnQKC#RKD-IWw6RZZHDS3ldM7suA">Download all models here</a>. You should do this if you want to play the weaker A.I.s (1-11). This is what the GUI looks like when it defeated the simple rule based A.I. we are required to write for the project (that I was playing for with the black stones).  

![Image of the GUI](images/defeated-simple-ai.png)

Note: If you want to play against the A.I.s, you need the required dependencies (tensorflow). If you are not on linux, you must recompile the nptrain library by executing `python3 build nptrainsetup.py` in your terminal and copying the respective file from the build directory. You may also have to change the file paths for all the code that loads files if you are on Windows. (remember to escape the backslashes in the fiile extention)

#### Training
For training, the batch size is set to be 32 (not sure if that is good or bad) and the metrics from training are shown. Note that the jump in the graph is there as I have forgot to rotate the boards during training, causing the first batches of training to be not as effective.

The accuracy of the policy prediction is given:  
![Image of policy accuracy](images/acc.png)  

The losses for the training (P+V), policy and value components are shown below:  
![Image of total loss](images/loss.png)  
![Image of policy loss](images/ploss.png)  
![Image of value loss](images/vloss.png)  
  
A problem with this model is the value network. It seems the value network always tends to predict very high numbers for the value, like the agent is always close to winning even in very early positions. This can be seen by the biases on the value layer and how they are always positive.  

![Image of value layer](images/val-layer.png)  

This results in certain positions where the A.I. may have a chance to win, but decides not to. I think this phenomenon can be attributed to the fact that the value it predicts for certain other positions are so close to 1 already and the difference between actually winning, (obtaining a reward of 1) and not winning and entering a position where the value is say 0.95 is too small to be relevent. This can be improved in the future by using a discounting factor less than 1.

