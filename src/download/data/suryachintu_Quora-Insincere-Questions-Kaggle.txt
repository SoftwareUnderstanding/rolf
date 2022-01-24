## Predicting Insincere Questions on Quora

This blog post is about the challenge that is hosted on kaggle [Quora Insincere Questions](https://www.kaggle.com/c/quora-insincere-questions-classification/overview). 

This post is divided into five parts:

1. Overview
2. Evaluation Metrics
3. Basic EDA and Data Preprocessing
4. Base Line Model
5. Deep Learning Models
6. Conclusion

Lets get started!

![GIF](https://media1.tenor.com/images/7dcc0b5a2c64d741b6edd12a88738cf9/tenor.gif?itemid=4767352)

### 1. Overview:
Quora is a platform that empowers people to learn from each other. One can go to Quora and ask their questions and get answers from others. But some questions asked by users may be toxic and contain divisive content. The aim of the competition is to tackle these situations.

The dataset is a csv file you can download it from [here](https://www.kaggle.com/c/quora-insincere-questions-classification/data).

<b>qid</b>: Question ID

<b>question_text</b>:  Question text.

<b>target</b>: target whether the question is sincere or not. if question is insincere then target is 1 else 0.


### 2. Evaluation Metrics
F1-Score = 2 x (precision x recall) / (precision + recall)

<b>Precision</b>: The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.

<b>Recall</b>: The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.

### 3. Basic EDA and Data Preprocessing

You can find the notebook [here](notebooks/QuoraInsincereQuestions-EDA.ipynb) for more detailed analysis. 


```python
# load the dataset
train_df = pd.read_csv('train.csv')
```
<img src="assets/dataset.png"/>

#### 3.a Target Label Distribution

<img src="assets/distribution.png"/>

The above plot clearly shows that the dataset is a imbalanced dataset with over 12,25,312 questions are sincere with target as 0 and 80810 are marked as insincere with target label as 1

#### 3.b Text Analysis

Finding the length of each question text in dataset

```python
# calculate length of question text
train_df['length'] = train_df['question_text'].apply(lambda x: len(x))
```

```python
print("Question text")
print('='*25)
print(x['question_text'].values[0])
print('='*25)
print("Target for this question:", x['target'].values[0])
```
<img src="assets/max_len.png"/>


<br/>

This question is has lot of math operations like powers mulitplications and all and the interesting part is question has been marked as insincere.

<br/>


```python
x = train_df[train_df['length'] == train_df['length'].min()]
print("Question text")
print('='*25)
print(x['question_text'].values[0])
print('='*25)
print("Target for this question:", x['target'].values[0])
```

<img src="assets/min_len.png"/>

<br/>
This question has no text. Interestingly this has been marked as <b>insincere</b>. since there is no point of keeping this point in the dataset we can remove this.
<br/>

Lets check distribution of length feature for sincere and insincere questions.


<img src="assets/length_dist.png"/>

Length seems to be not much useful from the above plot.


#### Lets check bad words count in a question

You can find the list of words in this repo https://github.com/RobertJGabriel/Google-profanity-words/blob/master/list.txt

```python
bad_words_df = pd.read_csv('bad_words_list.txt')
bad_words_df.head()
```
<img src="assets/bad_words_df.png"/>

Code for extracting bad words from question text

```python
# extract bad words from question text
bad_words = list(bad_words_df['word'].values)

from tqdm import tqdm

def count_bad_words(question_texts):
    b_count = []
    for question in tqdm(question_texts):
        b_count.append(sum([1 for w in bad_words if w.lower() in question.lower()]))
    return b_count

train_df['bad_words_count'] = count_bad_words(train_df['question_text'].values)

print(train_df['bad_words_count'].min(), train_df['bad_words_count'].max())
```

Output : 0, 9

Distribution of bad_words_count for sincere and insincere questions

<img src="assets/bad_words_dist.png"/>

Looks like this feature is not much useful.

Similarly we can find many other features like Country Count, Sentence Count, Word Count, Stop Word Count, Punctuation count, Average word length and code for all these feature is available in this [notebook](notebooks/QuoraInsincereQuestions-EDA.ipynb) but none of these features found helpful.

#### 3.c Text Preprocessing

You can find the notebook [here](notebooks/QuoraInsincereQuestions-Text-Preprocessing.ipynb)

We will be using [GloVe](https://nlp.stanford.edu/projects/glove/) for obtaining vector representation of a word as this is useful for our Deep learning model which will be covered at the end.

Code for loading the glove vector representations

```python
def get_coefs(word,*arr): 
    return word, np.asarray(arr, dtype='float32')

def load_embed(file):    
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))
    return embeddings_index

%%time
embed_glove = load_embed('glove.840B.300d.txt')
```
Once we have loaded Glove vector representations into dictionary we can access vector representations of any word by embeddings_index[word]. So embeddings_index is generally a dictionary containing key as a word and its value is a vector representation.

```python

def build_vocab(texts):
    """
    Helper function to create a vocab count for our data
    """
    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

def check_coverage(vocab, embeddings_index):
    """
    Helper function to check the code coverage with embedding matrix
    """
    known_words = {}
    unknown_words = {}
    nb_known_words = 0
    nb_unknown_words = 0
    for word in vocab.keys():
        try:
            known_words[word.strip()] = embeddings_index[word.strip()]
            nb_known_words += vocab[word]
        except:
            unknown_words[word] = vocab[word]
            nb_unknown_words += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(known_words) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))
    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]

    return unknown_words
```

So lets start by checking code coverage for our raw question text'

```python
%%time
vocab = build_vocab(train_df['question_text'])
oov_glove = check_coverage(vocab, embed_glove)
```
Output<br/>

Glove : <br/>
Found embeddings for 33.02% of vocab<br/>
Found embeddings for  88.15% of all text<br/>

We can only get vector representations for 33% of vocab(unique words) and 88.15% of all text.

Lets check what are missing in this

```python
print(oov_glove[:10])
```
Output:

[('India?', 16384),
 ('it?', 12900),
 ("What's", 12425),
 ('do?', 8753),
 ('life?', 7753),
 ('you?', 6295),
 ('me?', 6202),
 ('them?', 6140),
 ('time?', 5716),
 ('world?', 5386)]
 
 
As we can see from the above cell lot of text processing needs to be done. India? has appeared 16,384 times in our question texts.

Next step is to clean the raw questions by lower casing the text, expanding contractions and removing punctuations and check if our we can increase our vocab coverage.

```python
train_df['lowered_question'] = train_df['question_text'].apply(lambda x: x.lower())
# expanding code contractions 
# refer to https://gist.github.com/nealrs/96342d8231b75cf4bb82
def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text)

train_df['cleaned_text'] = train_df['lowered_question'].apply(lambda x: expandContractions(x))

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', 
          '*', '+', '\\', '•',  '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', 
          '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢',
          '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕',
          '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', 
          '∞','∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 
          'ï', 'Ø', '¹', '≤', '‡', '√', '∞', 'θ', '÷', 'α', '•', 'à', '−', 'β', '∅', '³', 'π', '‘','₹', 
          '´', "'", '°', '£', '€', '×', '™','√','²','—–','&','…', "’", "“", "”", "#", "{", "|", "}", "~"]
          
def remove_punctuations(text):
    for p in unknown_punctuations:
        text = text.replace(p, ' ')
    for p in known_puncts:
        text = text.replace(p, ' ' + p + ' ')
    return text
    
train_df['final_cleaned_text'] = train_df['cleaned_text'].apply(lambda x: remove_punctuations(x))

vocab = build_vocab(train_df['final_cleaned_text'])

oov_glove = check_coverage(vocab, embed_glove)

```
Output:

Glove : <br/>
Found embeddings for 69.57% of vocab<br/>
Found embeddings for  99.58% of all text<br/>

Now the vocab coverage is increased from 33% to 69.57% and covers 99.58% of all text. So removing punctuations, lower casing and expanding contractions increased vocab coverage.

We can further increase the coverage by corecting the spellings as there are lot of spelling mistakes in question text. You can find detailed code on this [notebook](notebooks/QuoraInsincereQuestions-Text-Preprocessing.ipynb)   


### 4. Base Line Model

You can find the notebook [here](notebooks/QuoraInsincereQuestions-ML-Models.ipynb)

We will be using Naive Bayes as our base line model with TFIDF vectorized words as Naive Bayes works pretty fast and can be used as base line model for many projects.

Complete code can be found in this [notebook](notebooks/QuoraInsincereQuestions-ML-Models.ipynb)
```python
%%time
# Estimator as MultinomialNB for GridSearchCV
mnb = MultinomialNB()
# Estimator MultinomialNB
# param_grid: Additive (Laplace/Lidstone) smoothing parameter : (10^(-10) to 10^(10))
# cv: Cross validation with 3 fold
# scoring: perform cross validation based on f1
# verbose: for debugging
clf = GridSearchCV(estimator=mnb, param_grid={'alpha': list(map(lambda x: 10**x , range(-10,10))) }, cv=3,
                   scoring='f1',verbose=10, n_jobs=-1)
# Fit the train samples
clf.fit(tfidf_train_vect, y_train.values)

print("TFIDF train f1 score:", f1_score(y_pred=clf.best_estimator_.predict(tfidf_train_vect), \
                                      y_true=y_train.values))
print("TFIDF test f1 score:", f1_score(y_pred=clf.best_estimator_.predict(tfidf_test_vect), \
                                     y_true=y_test.values))
                                     
```
Output:

TFIDF train f1 score: 0.7238521247204315<br/>
TFIDF test f1 score: 0.5829645584491874

By using Naive Bayes Classifier we are able to get 0.583 F1-score on our test set as this a pretty good score we can use it as base line model.

### 5. Deep Learning Model

You can find the notebook [here](notebooks/Quora-LSTM.ipynb)

We will be using Bidirectional LSTM with a [Attention Layer]() folowed by a dense layer of 64 units and another dense layer of 32 units with [elu](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#elu) activation function for our model. Attention Layer is based on the Dzmitry Bahdanau research paper https://arxiv.org/pdf/1409.0473.pdf and custom keras implementation is taken from kaggle kernel https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043 and detailed explaination will be found in this [notebook](notebooks/AttentionLayerDemo.ipynb), in this notebook there is a detailed explanation of input shapes and output shapes at of each operation used for the implementation.

```python
model = Sequential()
model.add(embedding_layer)
model.add(Bidirectional(LSTM(64, dropout=0.25, recurrent_dropout=0.25, return_sequences=True)))
model.add(Attention(MAX_SEQUENCE_LENGTH))
model.add(Dense(64, activation='elu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='elu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=[get_f1])

model.summary()
```
<img src='assets/model.png'/>

```python
model.fit(x_train, y_train,
          batch_size=2048*2,
          epochs=25,
          validation_split=0.3, callbacks=callbacks_list)
score, acc = model.evaluate(x_val, y_val,
                            batch_size=2048)

print('Test Loss:', score)
print('Test F1 Score:', acc)
```
Output:

391836/391836 [==============================] - 30s 77us/step<br/>
Test Loss: 0.10312879059361556<br/>
Test F1 Score: 0.6746514823591312<br/>

Lets check the model loss plot on train and validation sets

<img src='assets/loss.png'/>


After 7 epochs the training loss has decreased constantly but the validation loss does not decrease much and fluctuated from 0.11 to 0.12 . So we keep our checkpoint of the at epoch 7 and save the model.

Lets check the model f1-score plot on train and validation sets

<img src='assets/f1score.png'/>

From the abpve figure by looking at the f1 score after 7th epoch validation set f1 score became amost constant has no significant improvement and training set f1 score increased indicating the model is overfitting.


### 6. Conclusion

With bidirectional lstm using attention layer we are able to get 0.675 f1 score on our test set. We can improve the score by doing more text processing fixing the spell corrections and handling symbols and other special characters, adding more LSTM layers etc.


### References

1. https://www.kaggle.com/c/quora-insincere-questions-classification
2. https://www.appliedaicourse.com/course/11/Applied-Machine-learning-course
3. keras Attention layer implentation https://www.kaggle.com/qqgeogor/keras-lstm-attention-glove840b-lb-0-043
4. https://colah.github.io/posts/2015-08-Understanding-LSTMs/
5. https://machinelearningmastery.com/gentle-introduction-long-short-term-memory-networks-experts/