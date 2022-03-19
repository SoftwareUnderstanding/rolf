# Bert_Telugu_Ner
Finding the named entities from the sentences of telugu language trained the model using Googles BERT.
## ========== TELUGU_NER ===========

### Objective :
	 The main objective of this system is to predict the different named entities from the sentence
   	 on the telugu language using SOTA(State of the ART) multicased model google "BERT".

### Process :

	 - First I had collected the data from the <code> https://github.com/anikethjr/NER_Telugu <code>
	   which is collected from the telugu newspaper websites.
	 - Then I used the google BERT model trained on the English NER from <code>
	   https://github.com/kyzhouhzau/BERT-NER<code>. which is trained on English sentences.
	 - Then I modified some of the code for giving the different form of input and usage of the 
	   multicased model of Google BERT. Here while raining the model the multicased model consists of 
	   very few words which it is trained on, So the remaing words which it isn't trained on are splitted 
	   in to tokens with telugu characters.
	 - Atlast it is sent to the BERT model in the form of sentences with steps which the BERT model will be
	   trained on. Then we predict the NER of TELUGU words.
	   Note: At present we are satisfied with the accuracy but need much data to be trained on to get better
		  accuracy.
		  
### How to use:
	- First download the multicased BERT model and change the path in the ipynb file and use google colab
	  to train the model, It will take nearly 1-2 hours for training then download those trained models
	  and use them with <code>telugu_ner.py<code> which is just a prediction file.
	  
### References:
+ [https://github.com/google-research/bert](https://github.com/google-research/bert)
+ [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
+ [https://github.com/kyzhouhzau/BERT-NER](https://github.com/kyzhouhzau/BERT-NER)
+ [https://github.com/anikethjr/NER_Telugu](https://github.com/anikethjr/NER_Telugu)
+ [http://lrec-conf.org/workshops/lrec2018/W11/pdf/2_W11.pdf](http://lrec-conf.org/workshops/lrec2018/W11/pdf/2_W11.pdf)
		
