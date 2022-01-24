# Machine Reading Comprehension using SQUAD v.1

![Reading Coprehension](/images/reading_comprehension.jpg)

## About Dataset:
Data Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable. You can download this dataset here https://rajpurkar.github.io/SQuAD-explorer/

![Data Strucutre](/images/dataset.PNG)

**SQuAD 1.1:** The previous version of the SQuAD dataset, contains 100,000+ question-answer pairs on 500+ articles.

## Problem Statement
Predicting the right answer for the given question and context.

## Standford Attentive Reader
Implemented standford attentive reader model using keras.Please refer this [paper.](https://arxiv.org/pdf/1704.00051.pdf)

![Standford Attentive Reader](/images/model.JPG)

## BERT on SQUAD:

BERT, or Bidirectional Encoder Representations from Transformers, is a new method of pre-training language representations which obtains state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks.

Please refer this research paper. https://arxiv.org/abs/1810.04805.

**Disclaimer**

- Most of the code is taken from google-research github account
- The bert model is fine-tuned only.
- The code modified as per necesscity
- Used the bert base model with 110M parameters
- All the referance are mentioned in the referances section

**For ipynb notebook , please check the bert folder**

## Blog:
**I have written a detailed post regarding this on medium. You can read it here https://medium.com/@raman.shinde15/neural-question-and-answering-using-sqad-dataset-and-attention-983d3a1dd42c**


## Observations:

* Obtained micro f1_score of 40.33% on test data.
* Algined question embedding and f_exact match found to be the moset effective as mentioned in paper
* f1_score can be further improoved by adding Algined question embedding feature to context.
* Algined question embedding was omitted due to computational power limits
* To train on 1 epoch it took around hour without Algined question embedding
* Algined question embedding was omittited because, training on 1 epoch was taking more than 5 hours.
* Performance can be improoved further by considering:
    * All data points
    * Taking 128 units and 3 Layer of Bi_LSTM as mentioned in paper.
    * Considering Algined question embedding + f_exact together.
* Fine tuned Bert Uncased state of the art model to get the results.
* Bert model results are obtained using TPU provided by google

## Summary:
 
![Summary](/images/summary.PNG)
 
## Referances:

* The Stanford Question Answering Dataset by Rajpurkar https://rajpurkar.github.io/mlx/qa-and-squad/
* ReadingWikipedia to Answer Open-Domain Questions https://arxiv.org/pdf/1704.00051.pdf
* https://hanxiao.github.io/2018/04/21/Teach-Machine-to-Comprehend-Text-and-Answer-Question-with-Tensorflow/
* https://github.com/kellywzhang/reading-comprehension
* https://github.com/Shuang0420/Fast-Reading-Comprehension
* https://github.com/google-research/bert/blob/master/run_squad.py
* https://github.com/google-research/bert
* https://www.kaggle.com/lapolonio/bert-squad-forked-from-sergeykalutsky/code


----------------------------------------------------END------------------------------------------------------


