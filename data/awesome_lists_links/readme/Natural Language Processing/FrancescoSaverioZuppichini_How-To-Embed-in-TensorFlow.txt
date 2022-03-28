
# How to Embed words in Tensorflow
*Three different ways*

Today we are going to see how to create embeddings
using TensorFlow.

*Updated to tf 1.9*

(Words embedding)[https://en.wikipedia.org/wiki/Word_embedding] is a way to represent words by creating high dimensional vector space in which similar words are close to each other.

Long story short, Neural Networks work with numbers so you can't just throw words in it. You **could one-hot encoded** word but you will lose any relation between them. So you need to embed them.

Usually, almost always, you place your Embedding layer in-front-of your neural network.

## Preprocessing

Usually, you have some text files and you build a vocabulary after tokenizing the text, something similar to the following code


```python
import tensorflow as tf
import numpy as np
import pprint
import re
```

    /usr/local/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters



```python
text = "Quo usque tandem abutere, Catilina, patientia nostra? Quamdiu etiam furor iste tuus nos eludet? Quem ad finem sese effrenata iactabit audacia?"

tokenized = re.sub('[,?.]','', text).lower().split(' ') #Let's tokenize our text by just take each word
vocab = {k:v for v,k in enumerate(np.unique(tokenized))}
```

This is a very easy example, you usually need some more preprocessing to open multiple text source in parallel, create tokens and create the vocab. For example, you could use PySpark to efficiently preprocess text.

Let's print the vocabulary


```python
pprint.pprint(vocab)
```

    {'abutere': 0,
     'ad': 1,
     'audacia': 2,
     'catilina': 3,
     'effrenata': 4,
     'eludet': 5,
     'etiam': 6,
     'finem': 7,
     'furor': 8,
     'iactabit': 9,
     'iste': 10,
     'nos': 11,
     'nostra': 12,
     'patientia': 13,
     'quamdiu': 14,
     'quem': 15,
     'quo': 16,
     'sese': 17,
     'tandem': 18,
     'tuus': 19,
     'usque': 20}


Now we need to define the Embedding size, so the dimension of each vector, in our case 50, and the vocabulary length


```python
EMBED_SIZE = 50
VOCAB_LEN = len(vocab.keys())

print(VOCAB_LEN)
```

    21


We know need to define the ids of the words we want to embed. Just for example, we are going to take *abutere* and *patientia*


```python
words_ids = tf.constant([vocab["abutere"], vocab["patientia"]])
```

Just to be sure you are following me. `words_ids` represent the ids of some words in a vocabolary. A vocabolary is a map from words (tokens) to ids. 

**WHY** we have to do that? Neural Networks work with number, so we have to pass a **number** to the embedding layer

## 'Native' method

To embed we can use the low-level API. We first need to define a matrix of size `[VOCAL_LEN, EMBED_SIZE]` (20, 50) and then we have to tell TensorFlow where to look for our words ids using `tf.nn.embedding_lookup`.

`tf.nn.embedding_lookup` creates an operation that retrieves the rows of the first parameters based on the index of the second.


```python
embeddings = tf.Variable(tf.random_uniform([VOCAB_LEN, EMBED_SIZE]))
embed = tf.nn.embedding_lookup(embeddings, words_ids)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(embed))
```

    [[0.2508639  0.6186842  0.04858994 0.5210395  0.46944225 0.93606484
      0.31613624 0.37244523 0.8245921  0.7652482  0.05056596 0.82652867
      0.637517   0.5321804  0.84733844 0.90017974 0.41220248 0.659974
      0.7645968  0.5598999  0.40155995 0.06464231 0.8390876  0.139521
      0.23042619 0.04655147 0.32764542 0.80585504 0.01360166 0.9290798
      0.25056374 0.9695363  0.5877855  0.9006752  0.49083364 0.5052364
      0.56793296 0.50847435 0.89294696 0.4142543  0.70229757 0.56847537
      0.8818027  0.8013681  0.12879837 0.75869775 0.40932536 0.04723692
      0.61465013 0.97508   ]
     [0.846097   0.8248534  0.5730028  0.32177114 0.37013817 0.71865106
      0.2488327  0.88490605 0.6985643  0.8720304  0.4982674  0.75656927
      0.34931898 0.20750809 0.16621685 0.38027227 0.23989546 0.43870246
      0.49193907 0.9563453  0.92043686 0.9371239  0.3556149  0.08938527
      0.28407085 0.29870117 0.44801772 0.21189022 0.48243213 0.946913
      0.40073442 0.71190274 0.59758437 0.70785224 0.09750676 0.27404332
      0.4761486  0.64353764 0.2631061  0.19715095 0.6992599  0.72724617
      0.27448702 0.3829409  0.15989089 0.09099603 0.43427885 0.78103256
      0.30195284 0.888047  ]]


## Keras Layer

I am not a fan of keras, but something it can be useful. You can use a stand-alone layer to create the embeddings


```python
embeddings = tf.keras.layers.Embedding(VOCAB_LEN, EMBED_SIZE)
embed = embeddings(words_ids)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(embed))
```

    [[0.9134803  0.36847484 0.51816785 0.19543898 0.07610226 0.8685185
      0.7445053  0.5340642  0.5453609  0.72966635 0.06846464 0.19424069
      0.2804587  0.77481234 0.7343868  0.16347027 0.56002617 0.76706755
      0.16558647 0.6719606  0.05563295 0.22389805 0.47797906 0.98075724
      0.47506428 0.7846818  0.65209556 0.89036727 0.14960134 0.8801923
      0.23688185 0.70695686 0.59664845 0.6206044  0.69665396 0.60709286
      0.42249918 0.7317171  0.03822994 0.37915635 0.60433483 0.4168439
      0.5516542  0.84362316 0.27857065 0.33540523 0.8601098  0.47720838
      0.9827635  0.09320438]
     [0.27832222 0.8259096  0.5726856  0.96932447 0.21936393 0.26346993
      0.38576245 0.60339177 0.03083277 0.665465   0.9077859  0.6219367
      0.5185654  0.5444832  0.16380131 0.6688931  0.82876015 0.9705752
      0.40097427 0.28450823 0.9425919  0.50802815 0.02394092 0.24661314
      0.45858765 0.7080616  0.8434526  0.46829247 0.0329994  0.10844195
      0.6812979  0.3505745  0.67980576 0.71404254 0.8574227  0.40939808
      0.8668809  0.58524954 0.52820635 0.31366992 0.05352783 0.8875419
      0.04600751 0.27407455 0.6398467  0.74402344 0.9710648  0.5717342
      0.78711486 0.9209585 ]]


## Tensorflow Layers
Since Tensorflow is a mess, there are always several ways to do the same things and it is not clear which one you should use, so no surprise there is also an other function to create embeddings. 


```python
embed = tf.contrib.layers.embed_sequence(ids=words_ids, vocab_size=VOCAB_LEN, embed_dim=EMBED_SIZE)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(embed))
```

    [[ 0.11656719 -0.21488819  0.04018757 -0.18151578 -0.12417153 -0.00693065
       0.27286723  0.00712651 -0.05931629 -0.20677638  0.14741448 -0.24938995
      -0.21667814  0.09805503  0.2690411   0.20826831  0.19904876  0.08541816
       0.20128882  0.15323257 -0.0386056   0.03025511  0.11573204  0.2161583
      -0.02596462 -0.15845075 -0.26478297 -0.13366173  0.27797714 -0.08158416
      -0.25292248 -0.16360758 -0.1846793   0.2444193   0.13292032  0.15807101
       0.24052963 -0.0346185   0.02243239  0.2350963  -0.0260604   0.12481615
      -0.1984439   0.20924723 -0.00630271 -0.26579106  0.04491454  0.10764262
       0.170991    0.21768841]
     [-0.09142873 -0.25572282  0.2879894  -0.2416141   0.0688259  -0.06163606
       0.2885336  -0.19590749 -0.04164416  0.28198788  0.18056017 -0.03718823
      -0.09900685  0.14315534 -0.25260317 -0.00199199 -0.08959872  0.23495004
      -0.18945126 -0.16665417  0.18416747  0.05468053 -0.23341912  0.02287021
       0.27363363  0.07707322 -0.02453846  0.08111072  0.12435484  0.12095574
       0.2879583   0.12930956  0.09152126 -0.2874632  -0.26153982 -0.10861655
      -0.01751739  0.20820773  0.22776482 -0.17411226 -0.10380474 -0.14888035
       0.01492503  0.24255303 -0.10528904  0.19635591 -0.22860856  0.2117649
      -0.08887576  0.16184562]]


## Write your own module
You can create your custom class with the classic paradigm `__init__` + `__call__` to build it. 


```python
class Embedding():
    def __init__(self, vocab_size, emd_dim=50):
        self.vocab_size = vocab_size
        self.emd_dim = emd_dim
        self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.emd_dim]))

    def __call__(self, words_id):
        return tf.nn.embedding_lookup(self.embeddings, words_ids)
```


```python
embedding = Embedding(VOCAB_LEN, EMBED_SIZE)
embed = embeddings(words_ids)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(embed))
```

    [[0.19501674 0.954353   0.30957866 0.65923584 0.28241146 0.80623126
      0.46677458 0.5877205  0.25624812 0.03041542 0.24185908 0.8056189
      0.61915445 0.04368758 0.16852558 0.24910712 0.66250837 0.01929498
      0.82387006 0.8489572  0.3970251  0.8156922  0.5550339  0.39991164
      0.64657426 0.1980362  0.35962176 0.89992213 0.99705064 0.7636745
      0.5627477  0.09286976 0.12509382 0.9644747  0.3412783  0.3238287
      0.08844066 0.06885219 0.2377944  0.04519224 0.6535493  0.39360797
      0.69070065 0.44310153 0.58286166 0.32064807 0.9180571  0.47852004
      0.6686201  0.44279683]
     [0.0843749  0.77335155 0.14301467 0.23359239 0.77076364 0.3579203
      0.95124376 0.03154683 0.11837351 0.622192   0.44682932 0.4268434
      0.21531689 0.5922301  0.12666893 0.72407126 0.7601874  0.9128723
      0.07651949 0.7025702  0.9072187  0.5582067  0.14753926 0.6066953
      0.7564144  0.2200278  0.1666696  0.63408077 0.57941747 0.9417999
      0.6540415  0.01334655 0.8736309  0.4756062  0.66136014 0.12366748
      0.8578756  0.71376395 0.624522   0.22263229 0.35624254 0.00424874
      0.1616261  0.43327594 0.83355534 0.51896024 0.53433514 0.47303247
      0.7777432  0.4082179 ]]


## Pretrain Embed

You can benefit by using pre-trained words embedding since they can improve performance of your model. They are usually trained with enormous datasets, e.g. Wikipedia, with bag-of-words of skip-grammar models.

There are several of already trained embeddings, the most famous are (word2vec)[https://arxiv.org/abs/1301.3781] and (Glove)[https://nlp.stanford.edu/pubs/glove.pdf].

You can read more about training words embedding and how to train them from scratch in this (tensorflow tutorial)[https://www.tensorflow.org/tutorials/representation/word2vec] and this nice (article)[https://towardsdatascience.com/learn-word2vec-by-implementing-it-in-tensorflow-45641adaf2ac]

In the past, load into TensorFlow pre-trained word-embedding was not so easy. You had to download the matrix from somewhere and load into your program and maybe store it again. Now, we can use **TensorFlow Hub**. 

## TensorFlow Hub
You can use pre-trained word-embeddings easily with TensorFlow hub: a collection of the pre-trained module that you can just import in your code. Full list is [here](https://www.tensorflow.org/hub/modules/text)

Let's see it in action.

By the way, TensorFlow Hub is buggy and does not work well on Jupiter. Or at least on my machine, you can try to run it and see if it works for you. On PyCharm I have not any problem so I will copy and paste the output.

If you have the same problem please open an Issue on TensorFlow repository.


```python
import tensorflow_hub as hub

embed = hub.Module("https://tfhub.dev/google/Wiki-words-250-with-normalization/1")
embeddings = embed(['abutere','patientia']) # you don't even need your ids, you can just pass the tokens

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    print(sess.run(embeddings))
```

```[[ 0.01815782 -0.11736143 -0.01165295  0.10202649 -0.15543617  0.15407497
  -0.07163239  0.17790176 -0.18012059  0.13788173  0.1268917  -0.16257742
  -0.13499844  0.1701659   0.26743007 -0.15926908  0.10797838  0.05476788
  -0.13214736  0.15441446 -0.06283569 -0.22621672 -0.1865167   0.16468641
   0.22982213 -0.0526311  -0.03113755 -0.0620549  -0.18108433 -0.2304368
   0.10223346  0.24914174 -0.3410024  -0.03771868  0.07790497 -0.11456378
   0.17404653 -0.08352549  0.14454281 -0.2570933  -0.09521795  0.09158771
   0.04893332  0.12543996 -0.22482021 -0.1661538  -0.15144172 -0.32015714
   0.1434788   0.15947533]
 [ 0.14180504  0.10827583  0.22994775 -0.07204261  0.06852116 -0.19182855
   0.08858112 -0.08104847 -0.14247733 -0.15481676  0.05121056 -0.09623352
  -0.06614684 -0.24938259 -0.10725792 -0.12859781 -0.30667645  0.32551166
   0.01077376  0.03581994 -0.19545339 -0.12020707 -0.05161392  0.17412941
   0.19448319  0.19055173 -0.12632692 -0.00152465 -0.04917777 -0.20264266
  -0.00259693 -0.0659536   0.16362615 -0.11058624 -0.23266786  0.07123026
   0.08790443 -0.13033037 -0.12809968 -0.06643552  0.03927997  0.19020995
   0.26122165 -0.1893848  -0.09913436 -0.09246968  0.08428465 -0.01915
  -0.01001874  0.0972615 ]]```

You can also re-train them by just setting the `trainable` parameter of the `hub.Module` constructor to `True`. This is useful when you have a spefic domain text corpus and you want your embeddings to specialize on that. `hub.Module("https://tfhub.dev/google/Wiki-words-250-with-normalization/1", trainable=True)`

## Conclusions

Now you should be able to create your word embeddings layer efficiently in TensorFlow!

Thank you for reading,

Francesco Saverio Zuppichini
