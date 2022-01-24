Adding All Deep Learning Experiments that I'm doing. 
Mostly are experiments with fastai where I'm either reproducing a paper or testing out some functionality. 
The name of the repo doesn't stand anymore as I've added POCS with other libraries. 

Mostly are wrappers over the famous huggintransformers library which has pretrained models of XL-NET and BERT. 
Spacy has a decent Wrapper. 
And I'm using a BERT-NER somewhere. 


All experiments have been run on cloab. Storing the experiments and results here. 
1) REGULARIZING RNNS BY STABILIZING ACTIVATIONS David Krueger & Roland Memisevic
   https://arxiv.org/abs/1511.08400

2)ALL YOU NEED IS A GOOD INIT
 LSUV.ipynb
 LSUV-text.ipynb: I have been trying to implement the same for the text POC. 
 However fastai Hooks have some bug. 
 Inspired from Jeremy Howard's fastai course. Lesson over here: https://github.com/fastai/course-v3/blob/master/nbs/dl2/07a_lsuv.ipynb
 https://arxiv.org/pdf/1511.06422.pdf

3) Analyzing and Improving Representations with the Soft Nearest Neighbor Loss
 entanglement.ipynb 
 https://arxiv.org/pdf/1902.01889.pdf
 This implementation is inspired from this implementation here by the author: https://github.com/tensorflow/cleverhans/blob/master/cleverhans/loss.py
 Implemented a HookCallBack for the same. 
 Have to run experiments. 

4) spacy-transformers
 POC for transformers using spacy-transformers module. 
 IMDB 

5) BERT-NER
  Testing out an implenmentation of BERT for NER. 
  Wanted to see the performance on Location. 


6) location-wiki-india
 Created a fastai classifier for wikipedia pages about India. 
 Got to 88%. 
 Have to drive up the Language model. 


7) Roberta_HugginFace_FastAI.
Added POC for HugginFace with fastai. Notebook comprises of Roberta. 
Doesn't work directly for BERT. 
Just a POC. Needs to be explored further. 
Tutorial: https://towardsdatascience.com/fastai-with-transformers-bert-roberta-xlnet-xlm-distilbert-4f41ee18ecb2
Got it from Jeremy's twitter. 


I belive rest have self-explainatory names. 




