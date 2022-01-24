# Q-A-bot

Contains an end-to-end memory network trained on Facebook babi dataset. The model was trained on 100epochs as of now and model parameters and other tweaks are to be performed to achieve better results.

Full Details: https://research.fb.com/downloads/babi/

## Architecture used

![](model_architecture.png)

### For more details please check out the research paper below - 

Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus, "End-To-End Memory Networks", http://arxiv.org/abs/1503.08895

## Results

![](results.png)

## Sample output -

Note. For now one can only use words from the existing vocab.

     >>> my_story = "John left the kitchen . Sandra dropped the football in the garden ."
     
     >>> my_question = "Is the football in the garden ?"
     
     >>> Predicted answer is:  yes
     >>> Probability of certainty was:  0.9991091
