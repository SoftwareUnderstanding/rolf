# seq2seq-with-attention-OCR-translation
Machine translation project with a practical approach, the project will incorporate an open source OCR engine so we can feed images in the source language(Chinese)

#**1. Objectives**

The objective was to implement a whole NLP workflow and create a model that could be potentially used "in the wild".  
When I was just a newcomer to Beijing my biggest fear was to find something like this on my door:

![Image of Notification](https://github.com/IpastorSan/seq2seq-with-attention-OCR-translation/blob/master/data/descarga.jpg)

So, as part of my Master´s final thesis, I though about creating a Deep Learning system that could help me to find out what was going on just by taking a picture (For the record, I ended up being able to read those myself, but the fear of the 通知 was still there).

#**2. Process**

I decided to go for a sequence2sequence model implemented in **Tensorflow 2.0**

The repository consist on 3 models, in increasing order of complexity
1) Vanilla seq2seq
2) Multilayer seq2seq
3) Seq2seq with Attention (Bahdanau). The attention layer implementation comes from this [Tensorflow tutorial](https://www.tensorflow.org/tutorials/text/nmt_with_attention#write_the_encoder_and_decoder_model)

You will find a preprocessing script that is useful for preparing the data to feed into the model. The data comes from the sentence pairs of the Tatoeba project available here: [Manythings](http://www.manythings.org/anki/)

Preprocessing of Chinese text has some added difficulties, thats why I use [Jieba](https://github.com/fxsjy/jieba) to help with word-level tokenization and [HanziConv](https://github.com/berniey/hanziconv) to transform all the dataset into Traditional Chinese

Finally, the OCR (pretrained [Tesseract](https://github.com/tesseract-ocr/tesseract)) script is designed to put everything together, recover the pre-trained language model 
(note that I don't provide the weights as they are quite heavy, but I can send them over upon request), transform an image into a string and pre process it in such a way that the model can interpret it.

*If you have any feedback on implementation mistakes or ideas to improve, I will be happy to hear them!*

#**3. References**

• https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-inkeras.html

• https://towardsdatascience.com/day-1-2-attention-seq2seq-models-65df3f49e263

• https://towardsdatascience.com/light-on-math-ml-attention-with-keras-dc8dbc1fad39

• https://blog.floydhub.com/attention-mechanism/

• https://towardsdatascience.com/language-translation-with-rnns-d84d43b40571

• https://www.tensorflow.org/tutorials/text/nmt_with_attention#write_the_encoder_and_decoder_model

• https://www.kaggle.com/residentmario/seq-to-seq-rnn-models-attention-teacherforcing/notebook

• https://wanasit.github.io/attention-based-sequence-to-sequence-in-keras.html

• https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/

• https://machinelearningmastery.com/develop-word-embeddings-python-gensim/

• https://arxiv.org/pdf/1409.0473.pdf

• https://medium.com/analytics-vidhya/neural-machine-translation-using-bahdanauattention-mechanism-d496c9be30c3

• https://stackabuse.com/python-for-nlp-neural-machine-translation-with-seq2seq-inkeras/

• https://towardsdatascience.com/language-translation-with-rnns-d84d43b40571

• https://towardsdatascience.com/word-level-english-to-marathi-neural-machinetranslation-using-seq2seq-encoder-decoder-lstm-model-1a913f2dc4a7

• https://github.com/fxsjy/jieba

• Source Chinese Embedding: https://github.com/Kyubyong/wordvectors

• Source English Embedding: https://fasttext.cc/docs/en/english-vectors.html
