# Language_classifier_with_Naive_Bayes
This project is for Language Classification between the text from three different languages Slovak, Czech and English with the help of Naive Bayes algorithum

## Data 
The complete Dataset for training and validation of the Project is given in Data/Sentences folder. The folder contains files for each language these files consists of sentenses encoded in them. 

## Methodology:
The first step will be preprocssing  the text like, converting all text to lower case, removing punctuations, removing digits. Then dictonary will be created for all the unique words in it for each language. The Count vetorizer from sklearn is used to convert the words into vectors of size equivalent to the total number of unique words. Finally using Naive Bayes for Language classification. Further in order to improove the model, I extracted subwords like 'ea' ,'re',etc, from words for each language and used them instead of parent words which also reduces the class imbalance. The idea for taking subwords as features is taken from(https://arxiv.org/abs/1508.07909).


