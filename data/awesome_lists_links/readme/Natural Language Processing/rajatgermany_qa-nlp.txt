# Reading Comprehension QA system
In this repo, I built the the reading comprehension Question Answering system. I tried two approaches, for baseline I used machine learning algorithms and then I tried deep learning NLP techniques. 
For training the model I have used Stanford Question Answering Datatset (https://rajpurkar.github.io/SQuAD-explorer/)

## Technical details
**Embeddings**: To create the feature vectors I used the Infersent embedding model created by Facebook. The reason was solely based on using the sentence embeddings instead of word vectors because documents and questions are longer text and sentence embeddings will be better representation. Among sentence embeddings I choosen infersent because it has better performance to many downstream NLP tasks then others. This was experimented in this paper https://arxiv.org/abs/1705.02364. In this they have evaluated number of sentence embedding models and Infersent model has performed better then the rest.

### Machine Learning Model:
I created two features cosine similarity and euclidean distance between the question and each sentence of the document. To split the document into sentences I used TextBlob. Its has more accuracy then Nltk for the sentence splitting. The target variable is the sentence ID having the correct answer. I have fitted the data with  multinomial logistic regression and random forest. Among two Random forest has performed better. View the model in ml_model.ipynb

### Deep Learning Model: 
To improve upon machnine learning model results, I have implemented the deep learning NLP architecture Bidirectional Attention Flow for Machine Comprehension (BiDaf)model. I referenced this paper https://arxiv.org/abs/1611.01603 to create the architecture. I have used Keras Api's for building the model architecture. The model is trained with document and question vector built using the infersent. The model predicts answerBeginIndex, answerEndIndex in the document containing the answer. View the model in deep_learning_models.ipynb.

### Demo
To view the demo I have implement a api, run the file api.py . I attached the image below of the request from the postman

![alt text](./img/demo.png)


## Steps
- Clone the repo 
- Download Infersent - Follow the steps here https://github.com/facebookresearch/InferSent.
- Run api.py to see the demo or play with any of the model

## Future Work: 
Implement state of art model BERT and Albert https://arxiv.org/abs/1909.11942
