
<h1><center>READ: Recurrent Encoder Neural Language Model with Hierarichal Attention Decoder Fine tuning for Text Classification</center></h1>

This work is inspired by two recent advances in NLP:

1- ULMFiT: Transfer learning from pre-trained model for LM, fined tuned on NLP task

2- HATT: Hierarichal Attention Classifier

__What's in READ not in ULMFiT__
- Hierarichy: which is good for sentiment prediction
- Attention


__What's in ULMFiT not in READ__
- AWD-LSTM
- Pre-trained LM on wikitext, then IMDB
- LRFind
- Bidirectional

__References__
https://github.com/fastai/fastai/blob/master/courses/dl2/imdb.ipynb
https://arxiv.org/abs/1801.06146
https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf




```python
import numpy as np
import pandas as pd
from numpy import array
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os
import string
os.environ['KERAS_BACKEND']='tensorflow'

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import load_model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from pickle import dump
```

    Using TensorFlow backend.
    

# IMDb data

## Data loading


```python
from pathlib import Path

DATA_PATH=Path('./dat/')
DATA_PATH.mkdir(exist_ok=True)
#if not os.path.exists('./dat/aclImdb_v1.tar.gz'):
if not os.path.exists('./dat/aclImdb'):
    !curl -O http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz 
    !tar -xf aclImdb_v1.tar.gz -C {DATA_PATH}

```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 80.2M  100 80.2M    0     0  35.0M      0  0:00:02  0:00:02 --:--:-- 35.1M
    


```python
BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag
UNK_TOK = '_UNK_'
UNK_ID = 0 # 0 index is reserved for the UNK in both Keras Tokenizer and Embedding

PATH=Path('./dat/aclImdb/')
```


```python
CLAS_PATH=Path('./dat/imdb_clas/')
CLAS_PATH.mkdir(exist_ok=True)

LM_PATH=Path('./dat/imdb_lm/')
LM_PATH.mkdir(exist_ok=True)
```


```python
import re
re1 = re.compile(r'  +')
import html

def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x)) # Do not lower() so that capitalized words still hold a sentiment
```


```python
import numpy as np
CLASSES = ['neg', 'pos']#, 'unsup']

def get_texts(path):
    texts,labels = [],[]
    for idx,label in enumerate(CLASSES):
        for fname in (path/label).glob('*.*'):
            texts.append(fixup(fname.open('r', encoding='utf-8').read()))
            labels.append(idx)
    return np.array(texts),np.array(labels)
    #return texts, labels

trn_texts,trn_labels = get_texts(PATH/'train')
val_texts,val_labels = get_texts(PATH/'test')
```


```python
len(trn_texts),len(val_texts)
```




    (25000, 25000)




```python
for t in trn_texts[:10]:
  print(t)
  print('\n')
```

    As someone who was staggered at the incredible visuals of "Hero," I was anxious to see this film which was billed as being along the same lines, but better. It also featured an actress I like: Ziyi Zhang. Well, I was disappointed on both counts. I bought the DVD of this film sight-unseen, and that was a mistake. It was not better.
    
    I realize these flying-through-the-air martial arts films are pure fantasy but this story is stretched so far past anything remotely believable it just made me shake my head in disappointing disbelief. A blind woman defeating hundreds of opponents? Sorry, that's going a little far. Also, the major male character 'Jin" (Takeshi Kaneshiro) was so annoying with his dialog, stupid look on his face and stupid laugh, that he ruined the film, too.
    
    Despite the wonderful colors and amazing action scenes, this story - to me - just didn't have an appeal to make it a movie worth owning. This film is no "Hero" of mine!
    
    
    One of Scorsese's worst. An average thriller; the only thing to recommend it is De Niro playing the psycho. The finale is typically of this genre i.e. over-the-top, with yet another almost invincible, immune-to-pain villain. I didn't like the 60s original, and this version wasn't much of an improvement on it. I have no idea why Scorsese wasted his time on a remake. Then again, considering how bad his recent movies have been (I'm referring to his dull Buddhist movie and all the ones with his new favourite actress, the blond girl Di Caprio) this isn't even that bad by comparison. And considering Spielberg wanted to do the remake... could have been far worse.
    
    
    Oh man. If you want to give your internal Crow T. Robot a real workout, this is the movie to pop into the ol' VCR. The potential for cut-up lines in this film is just endless.
    
    (Minor spoilers ahead. Hey, do you really care if a film of this quality is "spoiled?") Traci is a girl with a problem. Psychology has developed names for it when a child develops a sexual crush on the opposite-sex parent. But this girl seems to have one for her same-sex one, and I don't think there's a term for that. It might be because her mother Dana is played by Rosanna Arquette, whose cute overbite, neo-flowerchild sexuality and luscious figure makes me forgive her any number of bad movies or unsympathetic characters. Here Dana is not only clueless to her daughter's conduct; she seems to be competing for the gold medal in the Olympic Indulgent Mother competition. 
    
    It's possible that Dana misses Traci's murderous streak because truth be told, Traci seems to have the criminal skills of a hamster. It's only because the script dictates so that she manages to pull off any kind of a body count.
    
    A particularly hilarious note in this movie is the character of Carmen, a Mexican maid who is described by Dana as around so long she's like one of the family although she dresses in what the director thought would say, "I just fell off the tomato truck from Guadalajara." Carmen is so wise to Traci's scheming, she might also wear a sign saying, "Hey, I'm the Next Victim!" Sure enough, Traci confronts Carmen as Carmen is making her way back from Mass, and bops her with one of those slightly angled lug wrenches that car manufacturers put next to your spare as a bad joke. I rather suspect than in real life those things are as useless as a murder weapon as they are for changing a tire. 
    
    In another sequence, Arquette wears a flimsy dress to a vineyard, under cloudy skies, talking to the owner. Cut to her in another flimsy dress under sunny skies, talking to the owner's brother. Then cut to her wearing the first dress, in the first location, under cloudy skies - but it's supposed to be later. You get the picture. We're talking really bad directing.
    
    As for skin, don't expect much, although Traci does own a nice couple of bikinis. 
    
    For those looking for a trash wallow, 8. For anybody else, 1/2.
    
    
    Stereotyped, derivative, unoriginal and boring Western. The two popular stars (Charlton Heston and James Coburn) both give performances that are far from their best, and justifiably so; they both have superficial roles and character traits stated mainly by dialogue. Heston is a sheriff who "liked the world better as it used to be before" and Coburn is an outlaw who "owes something to the man who locked him up and has to pay his debt". Additionally, Heston is so old that he has trouble riding a horse and Coburn is mean and tough but not as cold-blooded a killer as some of the minor villains. Apparently, the filmmakers couldn't come up with even ONE original idea about how to make this movie somewhat distinguished. (*1/2)
    
    
    With title like this you know you get pretty much lot of junk. Acting bad. Script bad. Director bad. Grammar bad.
    
    Movie make lot of noise that really not music and lot of people yell. Movie make bad racial stereotype. Why come every movie with black hero have drug addict? Why come hero always have to dance to be success? Why come famous rapper always have to be in dance movie? Why come letter "s" can't be in title?
    
    Hollywood need to stop dumb down audience and make movie that have people with brain who know how speak proper English.
    
    Do self favor and not go see.
    
    
    I was expecting a lot better from the Battlestar Galactica franchise. Very boring prequel to the main series. After the first 30 minutes, I was waiting for it to end. The characters do a lot of talking about religion, computers, programming, retribution, etc... There are gangsters, mafia types, who carry out hits. However, Caprica doesn't have the action of the original series to offset the slower parts.
    
    Let me give you some helpful advice when viewing movies: As a general rule, if there is a lot of excessive exploitive titillation, then you know the movie will be a dud. Caprica has lots of this. The director/writer usually attempts to compensate for his poor abilities by throwing in a few naked bodies. It never works and all it does is demean the (very) young actresses involved and I feel sorry for them. Directors/writers who do this should be banned from the business.
    
    If you want to be bored for an hour and a half, by all means, rent Caprica. There's (free) porn on the 'Net if you really want to see naked bodies. Otherwise, move along, nothing to see here.
    
    
    It WAS supposed to be the last Freddy movie (and it was for over 10 years)--you would think they would have tried to get a good movie done. But they turned out giving us the worst of the series (and that's saying a lot). The plot made no sense (I seriously can't remember it), all the main characters were idiots (you REALLY wanted them dead) and Freddy's wisecracks were even worse than usual. The only remotely good bit about this was a brief (and funny) cameo by Johnny Depp (the first "Nightmare" movie was his first).
    
    Also I originally saw it in a theatre were the last section (reaccounting Freddy's childhood) was in 3-D. Well--the 3-D was lousy--faded colors and the image going in and out of focus. Also the three flying skulls which were supposed to be scary (I think) had the opposite reaction from my audience. EVERYBODY broke down laughing. Looks even worse on TV in 2-D. Pointless and stupid--very dull also. Skip this one and see "Freddy vs. Jason" again.
    
    
    To be completely honest,I haven't seen that many western films but I've seen enough to know what a good one is.This by far the worst western on the planet today.First off there black people in the wild west? Come on! Who ever thought that this could be a cool off the wall movie that everyone would love were slightly, no no, completely retarded!Secondly in that day and age women especially black women were not prone to be carrying and or using guns.Thirdly whats with the Asian chick speaking perfect English? If the setting is western,Asia isn't where your going. Finally,the evil gay chick was too much the movie was just crap from the beginning.Now don't get me wrong I'm not racist or white either so don't get ticked after reading this but this movie,this movie is the worst presentation of black people I have ever seen!
    
    
    DO NOT WATCH THIS MOVIE IF YOU LOVED THE CLASSICS SUCH AS TOM WOPAT, JOHN SCHNEIDER, CATHERINE BACH, SORRELL BOOKE, JAMES BEST, DENVER PYLE, SONNY SHROYER, AND BEN JONES! THIS MOVIE WILL DISSAPPOINT YOU BADLY! First of all, this movie starts out with Bo and Luke running moonshine for Jesse. Bo and Luke would not do that ever on the real series! This movie portrays unimaginable characters doing things that never would have happened in the series. In the series, Uncle Jesse was honest, and law-abiding. In this movie, he is a criminal who is making moonshine and smoking weed with the governor of Georgia. Plus, if this was an extension adding on to the Dukes of Hazzard Reunion! and the Dukes of Hazzard in Hollywood, I have one question: HOW COULD UNCLE JESSE BE MAKING MOONSHINE WHEN HE DIED BEFORE THE DUKES OF HAZZARD IN Hollywood MOVIE? AND HOW IS BOSS HOGG ALIVE WHEN HE DIED BEFORE THE REUNION MOVIE IN 1997! MOVIE AND ROSCO RAN HAZZARD? IT SEEMS MAGICAL THAT THESE CHARACTERS CAME BACK TO LIFE, WHEN THEY HAVE BEEN DEAD FOR 11 AND 8 YEARS? If Hollywood really wanted to make a good movie, they should have brought back James Best, John Schneider, Tom Wopat, Ben Jones, and Catherine Bach like they did in 1997 and 2000 and made a family friendly movie with the living original characters that made the show what it was and still is compared to this disgusting, disgraced movie! If you want to see good Dukes movies, either buy the original series, or go out to walmart.com and buy the DVD set of 2 that includes the Reunion, and Dukes of Hazzard in Hollywood movies! They both star the original cast, and are family friendly! Don't waste your time on a movie that isn't worth the CD it's written on!
    
    
    The movie had a lot of potential, unfortunately, it came apart because of a weak/implausible story line, miscasting, and general lack of content/substance. One of the very obvious flaws was that Sean Connery, who played an Arab man, didn't know how to pronounce his own Arab name! This may seem a small flaw but it points to the seeming lack of effort in paying attention to details. The quality of acting was uniformly well below average. 
    
    Movie's solitary saving grace was the twist in the plot at the very end; and a french song (I don't recall the title). Overall, it was a pretty bad movie where Sean Connery was visibly miscast.
    
    
    


```python
np.random.seed(42)
trn_idx = np.random.permutation(len(trn_texts))
val_idx = np.random.permutation(len(val_texts))
```


```python
trn_texts = trn_texts[trn_idx]
val_texts = val_texts[val_idx]

trn_labels = trn_labels[trn_idx]
val_labels = val_labels[val_idx]
```

## Fit tokenizer


```python
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
VOCAB_SIZE = 60000
tokenizer = Tokenizer(nb_words=VOCAB_SIZE)
tokenizer.fit_on_texts(np.concatenate([trn_texts, val_texts]))
```

    /usr/local/lib/python3.6/dist-packages/keras_preprocessing/text.py:178: UserWarning: The `nb_words` argument in `Tokenizer` has been renamed `num_words`.
      warnings.warn('The `nb_words` argument in `Tokenizer` '
    


```python
# Insert UNK
tokenizer.word_index[UNK_TOK] = UNK_ID
```


```python
str2int = tokenizer.word_index
int2str = dict([(value, key) for (key, value) in str2int.items()])
```

# Utils


```python
GLOVE_DIR = "./dat/glove"
try:
    os.mkdir(os.path.join('./dat', 'glove'))
except:
    pass
!wget -P {GLOVE_DIR} https://github.com/ahmadelsallab/HierarichalAttentionClassifier_HATT_Sentiment/raw/master/data/glove/glove.6B.100d.txt

```

    --2019-05-07 11:16:32--  https://github.com/ahmadelsallab/HierarichalAttentionClassifier_HATT_Sentiment/raw/master/data/glove/glove.6B.100d.txt
    Resolving github.com (github.com)... 192.30.253.112
    Connecting to github.com (github.com)|192.30.253.112|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: https://raw.githubusercontent.com/ahmadelsallab/HierarichalAttentionClassifier_HATT_Sentiment/master/data/glove/glove.6B.100d.txt [following]
    --2019-05-07 11:16:33--  https://raw.githubusercontent.com/ahmadelsallab/HierarichalAttentionClassifier_HATT_Sentiment/master/data/glove/glove.6B.100d.txt
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 151.101.0.133, 151.101.64.133, 151.101.128.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|151.101.0.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 6339063 (6.0M) [text/plain]
    Saving to: ‘./dat/glove/glove.6B.100d.txt’
    
    glove.6B.100d.txt   100%[===================>]   6.04M  --.-KB/s    in 0.1s    
    
    2019-05-07 11:16:33 (63.6 MB/s) - ‘./dat/glove/glove.6B.100d.txt’ saved [6339063/6339063]
    
    


```python

def load_embeddings(embeddings_file):
    embeddings_index = {}
    f = open(embeddings_file)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

    f.close()

    print('Total %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.random.random((VOCAB_SIZE+1, EMBEDDING_DIM))
    for word, i in str2int.items():
        if i < VOCAB_SIZE:
          embedding_vector = embeddings_index.get(word)
          if embedding_vector is not None:
              # words not found in embedding index will be all-zeros.
              embedding_matrix[i] = embedding_vector    
    return embedding_matrix
```


```python
import nltk
nltk.download('punkt')
```

    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Unzipping tokenizers/punkt.zip.
    




    True



# Params


```python
LM_DATA_SIZE = 200000
LM_SEQ_LEN = 50
VOCAB_SIZE = 60000
MAX_SENT_LENGTH = LM_SEQ_LEN
MAX_SENTS = 15
MAX_NB_WORDS = VOCAB_SIZE
EMBEDDING_DIM = 100
#GLOVE_DIR = "./dat/glove"
```

# NLM

https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/

## Data preparation


```python
def prepare_lm_data(in_texts, seq_len, size):
    
    # organize into sequences of tokens
    length = seq_len + 1
    sequences = list()
    for i in range(length, len(in_texts)):
      if i < size:
        # select sequence of tokens
        seq = in_texts[i-length:i]
        if(len(seq) != 51):
          print(len(seq))
        # convert into a line
        #line = ' '.join(seq)
        # store

        sequences.append(seq)
        '''
        l = len(line.split())#len(tokenizer.texts_to_sequences(line)) 
        if  l!= 51:
          print(l)
        '''
        #print(line)
      else:
        break
        
    return sequences
```


```python
def binarize_lm_data(in_texts, tokenizer):
    
    sequences = []
    for t in in_texts:
      words_idx = []
      for w in t:
        if w in tokenizer.word_index:
          idx = tokenizer.word_index[w]
          if idx < VOCAB_SIZE:
            words_idx.append(idx)
          else:
            words_idx.append(UNK_ID)
        else:
          words_idx.append(UNK_ID)
          
      sequences.append(words_idx) 
    
    #sequences = [[tokenizer.word_index[w] for w in t] for t in in_texts]
    #return np.array(tokenizer.texts_to_sequences(in_texts))
    return np.array(sequences)
```


```python

texts = text_to_word_sequence(' '.join(list(trn_texts)))#list(trn_texts)# np.concatenate([trn_texts, val_texts])


text_sequences = prepare_lm_data(texts, LM_SEQ_LEN, size=LM_DATA_SIZE)


```


```python
for t in text_sequences[:10]:
  print(t)
```

    ['i', 'read', 'somewhere', 'that', 'when', 'kay', 'francis', 'refused', 'to', 'take', 'a', 'cut', 'in', 'pay', 'warner', 'bros', 'retaliated', 'by', 'casting', 'her', 'in', 'inferior', 'projects', 'for', 'the', 'remainder', 'of', 'her', 'contract', 'she', 'decided', 'to', 'take', 'the', 'money', 'but', 'her', 'career', 'suffered', 'accordingly', 'that', 'might', 'explain', 'what', 'she', 'was', 'doing', 'in', 'comet', 'over', 'broadway']
    ['read', 'somewhere', 'that', 'when', 'kay', 'francis', 'refused', 'to', 'take', 'a', 'cut', 'in', 'pay', 'warner', 'bros', 'retaliated', 'by', 'casting', 'her', 'in', 'inferior', 'projects', 'for', 'the', 'remainder', 'of', 'her', 'contract', 'she', 'decided', 'to', 'take', 'the', 'money', 'but', 'her', 'career', 'suffered', 'accordingly', 'that', 'might', 'explain', 'what', 'she', 'was', 'doing', 'in', 'comet', 'over', 'broadway', 'though']
    ['somewhere', 'that', 'when', 'kay', 'francis', 'refused', 'to', 'take', 'a', 'cut', 'in', 'pay', 'warner', 'bros', 'retaliated', 'by', 'casting', 'her', 'in', 'inferior', 'projects', 'for', 'the', 'remainder', 'of', 'her', 'contract', 'she', 'decided', 'to', 'take', 'the', 'money', 'but', 'her', 'career', 'suffered', 'accordingly', 'that', 'might', 'explain', 'what', 'she', 'was', 'doing', 'in', 'comet', 'over', 'broadway', 'though', 'it']
    ['that', 'when', 'kay', 'francis', 'refused', 'to', 'take', 'a', 'cut', 'in', 'pay', 'warner', 'bros', 'retaliated', 'by', 'casting', 'her', 'in', 'inferior', 'projects', 'for', 'the', 'remainder', 'of', 'her', 'contract', 'she', 'decided', 'to', 'take', 'the', 'money', 'but', 'her', 'career', 'suffered', 'accordingly', 'that', 'might', 'explain', 'what', 'she', 'was', 'doing', 'in', 'comet', 'over', 'broadway', 'though', 'it', "doesn't"]
    ['when', 'kay', 'francis', 'refused', 'to', 'take', 'a', 'cut', 'in', 'pay', 'warner', 'bros', 'retaliated', 'by', 'casting', 'her', 'in', 'inferior', 'projects', 'for', 'the', 'remainder', 'of', 'her', 'contract', 'she', 'decided', 'to', 'take', 'the', 'money', 'but', 'her', 'career', 'suffered', 'accordingly', 'that', 'might', 'explain', 'what', 'she', 'was', 'doing', 'in', 'comet', 'over', 'broadway', 'though', 'it', "doesn't", 'explain']
    ['kay', 'francis', 'refused', 'to', 'take', 'a', 'cut', 'in', 'pay', 'warner', 'bros', 'retaliated', 'by', 'casting', 'her', 'in', 'inferior', 'projects', 'for', 'the', 'remainder', 'of', 'her', 'contract', 'she', 'decided', 'to', 'take', 'the', 'money', 'but', 'her', 'career', 'suffered', 'accordingly', 'that', 'might', 'explain', 'what', 'she', 'was', 'doing', 'in', 'comet', 'over', 'broadway', 'though', 'it', "doesn't", 'explain', 'why']
    ['francis', 'refused', 'to', 'take', 'a', 'cut', 'in', 'pay', 'warner', 'bros', 'retaliated', 'by', 'casting', 'her', 'in', 'inferior', 'projects', 'for', 'the', 'remainder', 'of', 'her', 'contract', 'she', 'decided', 'to', 'take', 'the', 'money', 'but', 'her', 'career', 'suffered', 'accordingly', 'that', 'might', 'explain', 'what', 'she', 'was', 'doing', 'in', 'comet', 'over', 'broadway', 'though', 'it', "doesn't", 'explain', 'why', 'donald']
    ['refused', 'to', 'take', 'a', 'cut', 'in', 'pay', 'warner', 'bros', 'retaliated', 'by', 'casting', 'her', 'in', 'inferior', 'projects', 'for', 'the', 'remainder', 'of', 'her', 'contract', 'she', 'decided', 'to', 'take', 'the', 'money', 'but', 'her', 'career', 'suffered', 'accordingly', 'that', 'might', 'explain', 'what', 'she', 'was', 'doing', 'in', 'comet', 'over', 'broadway', 'though', 'it', "doesn't", 'explain', 'why', 'donald', 'crisp']
    ['to', 'take', 'a', 'cut', 'in', 'pay', 'warner', 'bros', 'retaliated', 'by', 'casting', 'her', 'in', 'inferior', 'projects', 'for', 'the', 'remainder', 'of', 'her', 'contract', 'she', 'decided', 'to', 'take', 'the', 'money', 'but', 'her', 'career', 'suffered', 'accordingly', 'that', 'might', 'explain', 'what', 'she', 'was', 'doing', 'in', 'comet', 'over', 'broadway', 'though', 'it', "doesn't", 'explain', 'why', 'donald', 'crisp', 'and']
    ['take', 'a', 'cut', 'in', 'pay', 'warner', 'bros', 'retaliated', 'by', 'casting', 'her', 'in', 'inferior', 'projects', 'for', 'the', 'remainder', 'of', 'her', 'contract', 'she', 'decided', 'to', 'take', 'the', 'money', 'but', 'her', 'career', 'suffered', 'accordingly', 'that', 'might', 'explain', 'what', 'she', 'was', 'doing', 'in', 'comet', 'over', 'broadway', 'though', 'it', "doesn't", 'explain', 'why', 'donald', 'crisp', 'and', 'ian']
    


```python
sequences = binarize_lm_data(text_sequences, tokenizer)

sz_limit = LM_DATA_SIZE# len(sequences)

# separate into input and output
sequences = array(sequences[:sz_limit])
X, y = sequences[:,:-1], sequences[:,-1]

#y = to_categorical(y, num_classes=vocab_size)
```


```python
print(X.shape)
print(y.shape)
```

    (199949, 50)
    (199949,)
    

## Model


```python
# define model
#model = Sequential()
#model.add(Embedding(vocab_size, 50, input_length=seq_length))
#model.add(LSTM(100, return_sequences=True))
#model.add(LSTM(100))

#GLOVE_DIR = "./dat/glove"

embeddings_file = os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')
embedding_matrix = load_embeddings(embeddings_file)
        
  
embedding_layer = Embedding(VOCAB_SIZE+1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=LM_SEQ_LEN,
                            trainable=True)
sentence_input = Input(shape=(LM_SEQ_LEN,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
l_word_enc = TimeDistributed(Dense(200))(l_lstm)
l_lstm_2 = LSTM(100)(l_word_enc)
#model.add(Dense(100, activation='relu'))
l_dense = Dense(100, activation='relu')(l_word_enc)
#model.add(Dense(vocab_size, activation='softmax'))
output = Dense(VOCAB_SIZE+1, activation='softmax')(l_lstm_2)
model = Model(sentence_input, output)
print(model.summary())
word_enc = Model(sentence_input, l_word_enc)

# compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```

    Total 7396 word vectors.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         (None, 50)                0         
    _________________________________________________________________
    embedding_1 (Embedding)      (None, 50, 100)           6000100   
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 50, 200)           120600    
    _________________________________________________________________
    time_distributed_1 (TimeDist (None, 50, 200)           40200     
    _________________________________________________________________
    lstm_1 (LSTM)                (None, 100)               120400    
    _________________________________________________________________
    dense_3 (Dense)              (None, 60001)             6060101   
    =================================================================
    Total params: 12,341,401
    Trainable params: 12,341,401
    Non-trainable params: 0
    _________________________________________________________________
    None
    


```python
# Mount GDrive
from google.colab import drive
drive.mount('/content/gdrive')
gdrive_path = 'gdrive/My Drive/Colab Notebooks/READ/dat'
#gdrive_path = './dat'
```

    Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3Aietf%3Awg%3Aoauth%3A2.0%3Aoob&scope=email%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdocs.test%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive.photos.readonly%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fpeopleapi.readonly&response_type=code
    
    Enter your authorization code:
    ··········
    Mounted at /content/gdrive
    


```python
# save the model to file
lm_model_file_name = 'lm_model.h5'
word_enc_model_file_name = 'word_enc_model.h5'

```


```python
load_prev_model = False
filepath = os.path.join(gdrive_path, lm_model_file_name)
if load_prev_model and os.path.exists(filepath):
  model = load_model(filepath)  
  #word_enc = load_model(os.path.join(gdrive_path, word_enc_model_file_name))  
  print('Existing LM loaded')
  
```


```python
import os
class SaveWordEncoder(Callback):
    '''
    def on_train_begin(self, logs={}):
        self.acc = []
    '''
    def on_epoch_end(self, batch, logs={}):
      word_enc.save(os.path.join(gdrive_path, word_enc_model_file_name))
        

word_enc_model_cbk = SaveWordEncoder()

```


```python
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
```


```python
callbacks_lst = [word_enc_model_cbk, checkpoint]
```


```python
'''
import keras.backend as K
K.clear_session()
model = load_model(filepath)
'''
```




    '\nimport keras.backend as K\nK.clear_session()\nmodel = load_model(filepath)\n'




```python

# fit model
model.fit(X, y, batch_size=128, epochs=100, callbacks=callbacks_lst)


```

    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    Epoch 1/100
    199949/199949 [==============================] - 316s 2ms/step - loss: 6.9614 - acc: 0.0833
    
    Epoch 00001: acc improved from -inf to 0.08335, saving model to gdrive/My Drive/Colab Notebooks/READ/dat/lm_model.h5
    Epoch 2/100
    199949/199949 [==============================] - 312s 2ms/step - loss: 6.5614 - acc: 0.1061
    
    Epoch 00002: acc improved from 0.08335 to 0.10615, saving model to gdrive/My Drive/Colab Notebooks/READ/dat/lm_model.h5
    Epoch 3/100
    199949/199949 [==============================] - 311s 2ms/step - loss: 6.4404 - acc: 0.1129
    
    Epoch 00003: acc improved from 0.10615 to 0.11290, saving model to gdrive/My Drive/Colab Notebooks/READ/dat/lm_model.h5
    Epoch 4/100
    199949/199949 [==============================] - 313s 2ms/step - loss: 6.3512 - acc: 0.1179
    
    Epoch 00004: acc improved from 0.11290 to 0.11791, saving model to gdrive/My Drive/Colab Notebooks/READ/dat/lm_model.h5
    Epoch 5/100
    199949/199949 [==============================] - 313s 2ms/step - loss: 6.2625 - acc: 0.1230
    
    Epoch 00005: acc improved from 0.11791 to 0.12299, saving model to gdrive/My Drive/Colab Notebooks/READ/dat/lm_model.h5
    Epoch 6/100
    199949/199949 [==============================] - 311s 2ms/step - loss: 6.2046 - acc: 0.1274
    
    Epoch 00006: acc improved from 0.12299 to 0.12745, saving model to gdrive/My Drive/Colab Notebooks/READ/dat/lm_model.h5
    Epoch 7/100
    199949/199949 [==============================] - 311s 2ms/step - loss: 6.1606 - acc: 0.1318
    
    Epoch 00007: acc improved from 0.12745 to 0.13184, saving model to gdrive/My Drive/Colab Notebooks/READ/dat/lm_model.h5
    Epoch 8/100
    199949/199949 [==============================] - 313s 2ms/step - loss: 6.1505 - acc: 0.1366
    
    Epoch 00008: acc improved from 0.13184 to 0.13664, saving model to gdrive/My Drive/Colab Notebooks/READ/dat/lm_model.h5
    Epoch 9/100
    199949/199949 [==============================] - 356s 2ms/step - loss: 6.1104 - acc: 0.1391
    
    Epoch 00009: acc improved from 0.13664 to 0.13913, saving model to gdrive/My Drive/Colab Notebooks/READ/dat/lm_model.h5
    Epoch 10/100
     41600/199949 [=====>........................] - ETA: 4:15 - loss: 6.0580 - acc: 0.1438


```python
#word_enc.save(os.path.join(gdrive_path, word_enc_model_file_name))
```

# HATT

## Data preparation


```python
from nltk import tokenize
def prepare_hier_data(in_texts, in_labels):
    
    reviews = []
    labels = []
    texts = []
    
    for idx in range(len(in_texts)):
        text = in_texts[idx]
        label = in_labels[idx]
        if label != 2:
          #print('Parsing review ' + str(idx))
          texts.append(text)
          sentences = tokenize.sent_tokenize(text)
          reviews.append(sentences)       
          labels.append(label)
    return reviews, labels
```


```python

from keras.utils.np_utils import to_categorical

def binarize_hier_data(reviews, labels, tokenizer):
    data_lst = []
    labels_lst = []
    for i, sentences in enumerate(reviews):
        data = UNK_ID * np.ones((MAX_SENTS, MAX_SENT_LENGTH), dtype='int32') # Init all as UNK
        for j, sent in enumerate(sentences):
            if j< MAX_SENTS:
                wordTokens = text_to_word_sequence(sent)
                k=0
                for _, word in enumerate(wordTokens):
                    if word in tokenizer.word_index:
                      if k<MAX_SENT_LENGTH and tokenizer.word_index[word]<MAX_NB_WORDS:
                          data[j,k] = tokenizer.word_index[word]
                          k=k+1
        data_lst.append(data)
        labels_lst.append(labels[i])
    data = np.array(data_lst)
    targets = np.array(labels_lst) 
    targets = to_categorical(np.asarray(targets))
    return data, targets
```


```python
train_texts_, train_labels_ = prepare_hier_data(trn_texts, trn_labels)
train_data, train_targets = binarize_hier_data(train_texts_, train_labels_, tokenizer)

```


```python
# 25k only are training out of 75k, becasue 50k are unsup --> label = 2
print('Shape of data tensor:', train_data.shape)
print('Shape of label tensor:', train_targets.shape)
```

## Split train/val


```python

VALIDATION_SPLIT = 0.2

indices = np.arange(train_data.shape[0])
np.random.shuffle(indices)
train_data = train_data[indices]
train_targets = train_targets[indices]
nb_validation_samples = int(VALIDATION_SPLIT * train_data.shape[0])

x_train = train_data[:-nb_validation_samples]
y_train = train_targets[:-nb_validation_samples]
x_val = train_data[-nb_validation_samples:]
y_val = train_targets[-nb_validation_samples:]

print('Number of positive and negative reviews in traing and validation set')
print(y_train.sum(axis=0))
print(y_val.sum(axis=0))
```

## Model


```python
from keras.models import load_model
def load_word_enc_model(word_enc_model_file_name):
    word_enc_model = load_model(word_enc_model_file_name)

    '''
    embeddings_file = os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')
    embedding_matrix = load_embeddings(embeddings_file)


    embedding_layer = Embedding(VOCAB_SIZE+1,
                              EMBEDDING_DIM,
                              weights=[embedding_matrix],
                              input_length=MAX_SENT_LENGTH,
                              trainable=True)
    sentence_input = Input(shape=(LM_SEQ_LEN,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
    l_word_enc = TimeDistributed(Dense(200))(l_lstm)
  

    word_enc_model = Model(sentence_input, l_word_enc)  
    '''
    
    print(word_enc_model.summary())
    return word_enc_model
```


```python

word_enc_model = load_word_enc_model(os.path.join(gdrive_path, word_enc_model_file_name))#model.load(word_enc_model_file_name)

```


```python

embeddings_file_name = os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')

# building Hierachical Attention network
embedding_matrix = load_embeddings(embeddings_file_name)

        
embedding_layer = Embedding(VOCAB_SIZE,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH,
                            trainable=True)

class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.he_normal()
        super(AttLayer, self).__init__(**kwargs)
    '''
    def build(self, input_shape):
        assert len(input_shape)==3
        self.W = self.init((input_shape[-1],1))
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!
    '''
    def build(self, input_shape):
        assert len(input_shape)==3
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name='kernel', 
                                      shape=(input_shape[-1], 1),
                                      initializer='uniform',
                                      trainable=True)
        super(AttLayer, self).build(input_shape)  # Be sure to call this at the end
    
    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))
        
        ai = K.exp(eij)
        weights = ai/tf.expand_dims(K.sum(ai, axis=1), 1)
        
        weighted_input = x*weights
        return tf.reduce_sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
'''
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
l_dense = TimeDistributed(Dense(200))(l_lstm)
'''
l_dense = word_enc_model(sentence_input)
l_att = AttLayer()(l_dense)
sentEncoder = Model(sentence_input, l_att)

review_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
l_dense_sent = TimeDistributed(Dense(200))(l_lstm_sent)
l_att_sent = AttLayer()(l_dense_sent)
preds = Dense(2, activation='softmax')(l_att_sent)
model = Model(review_input, preds)

word_enc_model.trainable = True

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
              
```


```python
model.summary()
```


```python
hatt_model = 'hatt_model.h5'
load_prev_model = False
filepath = os.path.join(gdrive_path, hatt_model)
if load_prev_model and os.path.exists(filepath):
    model = load_model(filepath) 
    print('HATT model loaded')
```


```python
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
```


```python
callbacks_lst = [checkpoint]
```


```python

NUM_EPOCHS = 100
BATCH_SIZE = 50
print("model fitting - Hierachical attention network")
model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=NUM_EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks_lst)
```

# Test


```python
test_texts_, test_labels_ = prepare_hier_data(val_texts, val_labels)
test_data, test_targets = binarize_hier_data(test_texts_, test_labels_, tokenizer)
```


```python

print('Shape of data tensor:', test_data.shape)
print('Shape of label tensor:', test_targets.shape)
```


```python
for i, rev in enumerate(test_texts_[:100]):
    print(rev)
    test_input = test_data[i].copy()
    test_input = np.reshape(test_input, (1,test_input.shape[0], test_input.shape[1]))
    prediction = model.predict(test_input)
    print('Prediction: ', prediction)
    sentiment = np.argmax(prediction)
    print('Sentiment: ' + str(sentiment))
```
