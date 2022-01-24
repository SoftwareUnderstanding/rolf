# Natural-Language-Processing
BERT, ULMFiT

This notebook is used for the medical note classification 2019 on Kaggle https://www.kaggle.com/c/medicalnotes-2019/overview
The 1st winning team got 0.76699 of accuracy. The classification I got using BERT model is 0.79. (This was temporiraly tested on a seperate hold out test set since the test set for the competition is not published. I will submit the result and report the result on the competition's test set later.) 

I used 2 models: BERT and ULMFiT, but there are others to try out (XLM, ALBERT, ...)
- Accuracy for BERT 0.79
- Accuracy for ULMFiT 0.73
Note that I did not do vigorous fine-tuning. 

## Some of the ideas to improve the model performance:
- BERT requires input tokens to be truncated to 512 (510 to be specific, excluding CLS and SEP tokens). Most of the clinical notes are more than 1000 words in length (before bert tokenization), and note that the subword tokenization aldo reduced the numbers of input words (as same words can be splitted to subwords). Clinical notes do have a lot of the subword (e.g. ##os, ##es). One way to make use of all data, maybe to 
  + Try classification task using either the first part, middle part, or last part to see which one is best. The intuiation is some time, info at the begining may be more important than the end, or vice versus.   
  + Instead of using one segmentation to work on, segment the text into many parts that maybe allow some overlap, then make classification and average the results. 

- For ULMFiT model: During the language model training, spacy tokenizer and ADW LSTM. We can try using transformer language models instead, and then passing the leanred encoder for the second classification networks. 



Dependency: fastai library
The codes is attributable to Maximilien Roberti (BERT) and fastai lecture (ULMFiT)
## These are good source for state of the art NLP self-learning. 
- https://www.kaggle.com/maroberti/fastai-with-transformers-bert-roberta
- https://www.youtube.com/watch?v=MDX_x6rKXAs&t=1377s
- https://www.youtube.com/watch?v=ycXWAtm22-w&t=2709s
- http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
- https://medium.com/@jonathan_hui/nlp-bert-transformer-7f0ac397f524
- https://towardsdatascience.com/bert-roberta-distilbert-xlnet-which-one-to-use-3d5ab82ba5f8
- https://towardsdatascience.com/comparing-transformer-tokenizers-686307856955
## And of course the papers:
- https://arxiv.org/abs/1706.03762
- https://arxiv.org/abs/1810.04805
- https://arxiv.org/abs/1906.08237

