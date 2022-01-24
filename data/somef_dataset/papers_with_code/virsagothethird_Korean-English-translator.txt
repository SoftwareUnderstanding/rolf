# Korean-English Translator


## Objective

The objective for this project was to create a Neural Machine Translator that could translate English phrases to Korean.

You're probably wondering: Google Translate's a thing, why not just use that?

I have lived in South Korea for the past 9 years, and when I first moved there, Google Translate was not very good at translating Korean-English and vice-versa. Having found this out after several attempts at translation, I ended up using my trusty dictionary when I encountered phrases that I did not understand, looking up each word individually. Unfortunately, this distrust of Google Translate persists today even though it has vastly improved since, and I continue to work inefficiently with the dictionary. So, in order to make my life a little easier, I decided to try and make my own translator.



## English- Korean Translations are difficult

Here's a quick look at Korean vs English:

#### 안녕하세요. 재 이름은 김주경입니다.  ==  Hello. My name is Joo Kyung Kim

The first thing that you notice is that the alphabet is completely different. That's not the main issue when translating, though. One of the biggest issues is the grammar difference.

If we were to take a Korean simple sentence...

#### 톰은 한국어 공부해

...and translate it word for word into English, we would get this:

#### Tom Korean studies

Korean follows a Subject-Object-Verb grammar structure compared to English's Subject-Verb-Object structure. Seeing as how this is a simple example, it's easy to understand what I am trying to convey with this word-for-word translation. However, as the complexity of the sentence increases, the complexity of the translation also increases accordingly as seen below:


![grammer](https://github.com/virsagothethird/Korean-English-translator/blob/master/korean_english_grammar.jpg)


The use of honorifics is also highly important in Korean. Depending on who I speak to, I will adjust my speech accordingly even if I was conveying the exact same message. These intricacies can prove to be quite difficult to pick up for a machine algorithm. These are just a few of the reasons why students in Korea sometimes struggle when learning English.



## Initial EDA

I started with a dataset of a little over 3000 English-Korean sentence pairs from http://www.manythings.org/anki/. I further enriched my dataset by using a custom webscraper that scraped through thousands of KPOP song lyrics and lyric translations from https://colorcodedlyrics.com/ and obtained an addictional 95,000 English-Korean sentence pairs.

Seeing the recent rise in KPOP internationally, I reasoned that the quality of song lyric translations would have risen in proportion to it's popularity as many more record labels now release official translations to their songs.

This left us with a total dataset size of **98,161 sentence pairs** after cleaning with an English vocabulary size of **12,251 unique words** and a Korean vocabulary size of **58,663 unique words**.

Looking at the top 20 words in our English vocabulary:

![bar](https://github.com/virsagothethird/Korean-English-translator/blob/master/top_20.png)

Most of these words are stop words. Not surprising as these are used in most sentences.



## Processing each sentence

In order to make the input sentences machine readable, they must be transformed into numerical values. Using the preprocess function from our .py file, we will add <start> and <end> tokens ot each sentence, lowercase, remove punctuations, and split words that need to be split. We then use the tokenize function to transform the sentences into vectors where each number in the vector corresponds to a unique word in the vocabulary.
  
  Hello everyone, how're you guys doing?
  <start> hello everyone how are you guys going <end>
  [ 1, 648, 192, 68, 17, 4, 428, 250, 2, 0, 0, 0, 0,…]

This input sequence will then be fed into our model.



## The model

The model we will be using is a Seq2Seq model made in Tensorflow. It is a simple Encoder-Decoder model that utilizes LSTM layers.

![model](https://github.com/virsagothethird/Korean-English-translator/blob/master/plots/aws_model.png)

Very simply, the Encoder reads in the input sequences and summarizes the information as internal state vectors(hidden state and cell state). The decoder then uses these vectors as initial states and starts to generalize the output sequences. Since the Encoder outputs 2D vectors and the Decoder expects 3D arrays, the RepeatVector layer is used to ensure that the output of the Encoder is repeated a suitable number of times to match the expected shape. The TimeDistributed-wrapped Dense layer  applies each slice of the sequence from the Decoder LSTM as inputs to the Dense layer so that one word can be predicted at a time.

As this model will be dealing with multi-class classification, we will be using Categorical crossentropy as our loss function.



## Results

![loss](https://github.com/virsagothethird/Korean-English-translator/blob/master/plots/loss_plot78528.png)
![acc](https://github.com/virsagothethird/Korean-English-translator/blob/master/plots/acc_plot78528.png)

We see that the loss on our training set decreases with more training but our loss on the test set increases incrementally over time. We also see that the accuracy on the training set increases while the accuracy on our test set stagnates at around 0.71. 

This may be telling us that instead of actually learning how to translate, it is slowly memorizing the sentence pairs.

Despite this we can try to make some translations on our flask app:

![badtranslation1](https://github.com/virsagothethird/Korean-English-translator/blob/master/plots/bad_translator.png)

This output a Korean sentence that translates to:
**Tom is a real man**

![badtranslation2](https://github.com/virsagothethird/Korean-English-translator/blob/master/plots/bad_translator2.png)

This outputs a Korean sentence that translates to:
**I love you I love you**



## Looking Forward

As we can see, the model is still very much in the toddler stages. There is much more room for growth. Additional hyperparameter tuning as well as going with a deeper neural network could help improve the performance of the model. More data (millions of sentence pairs) will also be a boon as these types of models tend to be very data hungry. 



## Resources
https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/
https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/
https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/
https://machinelearningmastery.com/encoder-decoder-long-short-term-memory-networks/
https://machinelearningmastery.com/introduction-neural-machine-translation/
https://www.tensorflow.org/tutorials/text/nmt_with_attention
https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
https://github.com/lukysummer/Machine-Translation-Seq2Seq-Keras
https://arxiv.org/pdf/1609.08144.pdf
