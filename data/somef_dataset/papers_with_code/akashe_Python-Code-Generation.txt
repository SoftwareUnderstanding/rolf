# Python Code Generation

#### Observations:
The code in the repository doesn't create the right code for any possible python question. It is trained on a small data and creates a good program related to the questions present in the data. However, it still doesn't understand indentation properly for very long programs. After a ':' model correctly predicts the amount of indentation for the line but fails to capture which statements to keep in which indent specially for a very long program.


#### Steps taken for data cleaning
 1. Remove all comments \# and '''...''': This is to reduce the vocab size and make the problem simpler for the model
 2. Replace tabs with "    ": This helps to keep same indentation scheme in the file. Specially for cases with indentation scheme as 4-3-2 spaces.
 3. Replacing multiple line declarations of variables: We use python's own [tokenizer](https://docs.python.org/3/library/tokenize.html). It was creating problems with multiline declarations.  
  

#### Steps for data processing:
1. Removing duplicate question answer pairs: Original data had many duplicate questions and python codes submitted by as same assignment by different team members. After removing duplicate pairs, the total unique question answer pair we about 3100 as compared to 4600+ original pairs.
2. Tokenization: As said earlier, we used python's own tokenizer. There was a problem with it though. It took strings like the ones present in print('Akash Kumar') as a seperate string token 'Akash Kumar'. This unnecessarily increase vocab size. So tokenized these strings as characters to increase models verstality.
3. Formatting with proper indentation: Data had multiple indentation schemes. We identify the indent required and finally replace it with corresponding '\t' to keep sequence length smaller.

#### Loss Function:
Primarily we used Cross Entropy as our loss function. We experimented with an [additional penalty](https://github.com/akashe/Python-Code-Generation/blob/main/Original_data_with_penalty.ipynb) for code that fails execution but:
1. The new penalty based loss function make the training really slow because for each output in a batch we had to execute the script to get the result.
2. Model didn't learn. Since there is no way for the parameters to find gradients wrt to actual execution of the scripts, we multiplied it as a separate constant to the loss value. This changes the gradient value and naturally it didn't work. But we tried to see if we can atleast have some rudimentary learning which we can adjust with the punishment_constant we chose as a hyperparameter.

#### Python Embeddings:
We created [python embeddings](https://github.com/akashe/Python-Code-Generation/blob/main/Python_Embeddings_on_CoNaLa_mined_data.ipynb) using [CONALA mined dataset](https://conala-corpus.github.io/). The dataset consists of 590763 snippets of python. We train Decoder only transformer architecture and train it in an autoregressive manner. The task is simple to predict the next word given an input token. We train embeddings for a total of 15018 tokens which we got after using pythons in built tokenizer on the CONALA mined dataset. 

#### Data Expansion:
In addition to 3100 examples from the original data we add additional 2800 examples from conala-train and conala-test datasets. The datasets are of same format with a natural language prompt for a python code and the corresponding python snippet.

#### Architecture:
Architecture is same as mentioned in the paper ["Attention is all you need."](https://arxiv.org/abs/1706.03762). It's an encoder decoder model with the natural language prompt processed by the encoder and code generation by the decoder using multi-headed and self attention.

#### Evaluation metrics:
We used Rouge-L metric which matches the longest subsequence. In code, there is a fixed structure with snippets following each other to build on previous snippets. In machine translation, the same words can come in the beginning and at the end to form the same meaning so n-grams based evaluation metric makes sense.
Since in the code, n-grams presenting anywhere doesn't make sense, we chose [ROUGE-L metric](https://www.aclweb.org/anthology/P04-1077.pdf). It gives score according to matching of the longest common subsequence in target and output codes. We get a maximum of 15.8 ROUGE-L score on the validation set. Refer [this file](https://github.com/akashe/Python-Code-Generation/blob/main/Conala_with_original_data_with_python_embeddings.ipynb) for code.

#### Experiments:
Here are the different experiments we did, and their corresponding files.

1. [Vanilla encoder-decoder architecture with word vocab](https://github.com/akashe/Python-Code-Generation/blob/main/Vanilla_Enocder_Decoder_Architecture.ipynb):
    The first experiment with a simple encoder-decoder architecture with python tokenizer and no pretrained embeddings.

2. [Vanilla encoder-decoder architecture with char vocab](https://github.com/akashe/Python-Code-Generation/blob/main/Vanilla_Enocder_Decoder_Architecture_with_character_wise_decoder_vocab.ipynb):
    In this file, we do similar things as above we just used a char vocab for the decoder. We realized, that decoder outputs didn't have space between statements like 'def gdc(x,y)'
   
3. [Penalty for wrong execution](https://github.com/akashe/Python-Code-Generation/blob/main/Original_data_with_penalty.ipynb):
   As discussed earlier, the model didn't train well with an extra execution penalty because it deflected gradients from their true direction. 

4. [Training python embeddings](https://github.com/akashe/Python-Code-Generation/blob/main/Python_Embeddings_on_CoNaLa_mined_data.ipynb):
    In this file, we trained decoder embeddings for python tokens using 590763 instances of mined data in conala-mined dataset. The embeddings along with their corresponding vocab are present in the data folder.

5. [Conala data with original data](https://github.com/akashe/Python-Code-Generation/blob/main/Conala_with_original_data.ipynb):
    Similar to details in 1. Here, we trained our model on more data from conala train and test files from CONALA dataset.    

6. [Conala data with original with python Embeddings](https://github.com/akashe/Python-Code-Generation/blob/main/Conala_with_original_data_with_python_embeddings.ipynb):
    We use this file to report our results. We train on total 5937 unique question answer pairs along with the pretrained embeddings we got from 4.
   
#### Example Outputs:
Check the [file here for example outputs](https://github.com/akashe/Python-Code-Generation/blob/main/data/example_output.txt) for better formatting. Refer [this file](https://github.com/akashe/Python-Code-Generation/blob/main/Conala_with_original_data_with_python_embeddings.ipynb) for code.

#### Attention graphs between text and python code 

Reference paper:
https://arxiv.org/pdf/2002.05442.pdf
