# NLP_Project_AI2_Reasoning_Challenge ( Currently Under Development)
## Branches (under development)
Current there are five branches corresponging to our five group members. Each branch will contain codes, papers to read and other materials. The following describs briefly the what tasks and files are included in each branch.
* arc-chi: a) test run an existting model, b) test a new background paragraph search method, 3)literature review.
* arc-fumin: a)crawling background knowledge from internet, b)literature review, c)potentially test run a language model.
* arc-hguan6: a)test run baseline model and error analysis, b)test new model, c) literature review.
* arc-manlin: a) variou scripts, b) model test, c) literature review
* arc-yilun: a)crawling background knowledge from internet, b)literature review, c)potentially test run a language model.

### Disclaimer: 
Every team member is contributing to this project actively on different aspects of this project and we could assume our members are comtributing EQUALLY in the effort to this project. And appreciations are expressed to those who work on this project.

## File Description
1. T5 model: t5_test.ipynb and t5_ARC.ipynb

2. BERT baseline model: arc_easy_BERT_base_model.ipynb and arc_challenge_BERT_base_model.ipynb

3. RoBERTa-base without/without knowl-edge: LSH_attention.ipynb

4. Report for the project: CSE_576_2020Spring_Project_ARC.pdf

## Dataset Description
The data set could found [Allen Institute for AI ARC](https://leaderboard.allenai.org/arc/submissions/public). The dataset contains 7,787 natural grade-school level multiple-choice SCIENCE questions. This dataset's level of difficulty requires far more powerful knowledge and reasoning capability than ever before datasets such SQuAD or SNLI. The data set has two partitions: EASY Set and CHALLENGE Set. And inside each set, it is also devided into train, test and development sets. A corpus is also given in the dataset which could be used as background inforamtion source. But the ARC challenge is not limited to this corpus knowledge and it could also be open book.

<b> Easy: </b>  
Easy-Train Set: 2251 questions  
Easy-Test Set: 2376 questions  
Easy-Development Set: 570 questions  

<b> Challenge: </b>  
The Challenge Set contains only questions answer incorrectly by both a retrieval-based algorithm and a word co-occurence algorithm.  

Challenge-Train Set: 1119 questions  
Challenge-Test Set: 1172 questions  
Challenge-Development Set: 299 questions  

<b> Reference: </b>  
P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord. 2018. Think you have solved question answering? Try ARC, the AI2 reasoning challenge. CoRR, abs/1803.05457.

<b> Example: </b>  
EASY:
Which technology was developed most recently?  
&nbsp; &nbsp; A. cellular telephone(correct)  
&nbsp; &nbsp; B. television  
&nbsp; &nbsp; C. refrigerator  
&nbsp; &nbsp; D. airplane  

CHALLENGE:
Which technology was developed most recently?  
&nbsp; &nbsp; A. cellular telephone  
&nbsp; &nbsp; B. television  
&nbsp; &nbsp; C. refrigerator  
&nbsp; &nbsp; D. airplane (correct)

## Baseline
We use the [AristoRoBERTaV7](https://leaderboard.allenai.org/arc/submission/blcotvl7rrltlue6bsv0) as our baseline model. Currently AristoRoBERTaV7 can achieve accuracy around 0.66 on the test set.The baseline model is based on the [RoBERTa-Large model](https://arxiv.org/abs/1907.11692) and submitted by the Aristo team at Allen Institute for AI in Aug 2019.

## Our Model 
Currently, the bigest change in models to tackle the ARC dataset lies in the fact that it is difficult to search the macthing paragraphs in the background knowledge due to the broad range of the questions from math, phasics, biology, chemstry,  and various science subjects. The search method in most models are based tf-idf matching the keyword in questions and answer. In our first round error analysis of the base model, we find many of the wrong answers are caused by unmatching background paragraph and ambiguous paragraph. We plan to adopt some innovative search methods which improve the search method. And we are still DEVELOPPING the method now. In regarding to the language model, we plan to adopt BERT or RoBERT. Content of our method will be updated soon...
## Existing Papers
1. Chen Zhu, Yu Cheng, Zhe Gan, Siqi Sun, Thomas Goldstein, and Jingjing Liu. Freelb: Enhanced adversarial training for language understanding. arXiv preprint arXiv:1909.11764, 2019. 
## Further Reading
1. [Transformer Visualization](http://jalammar.github.io/illustrated-transformer/)  
