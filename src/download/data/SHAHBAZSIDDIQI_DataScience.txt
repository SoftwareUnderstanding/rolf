# DataScience
The repository consist of multiple projects in various domains like NLP, Computer Vision, Graph Embedding, Speech Recognition. Apart from some of my work in form of code, I have also included some of my presentation for better understanding of my work along with the results obtained and my review on the algorithms. 

Natural Language Processing: 

Machine Translation from English to German: Neural machine translation from English to German and again back to English from German. dataset: http://www.manythings.org/anki/deu-eng.zip • Implemented Encoder- decoder based Sequence-to-Sequence (seq2seq) model on only 20k data out of over 150,000 data due to less computational power. • For the encoder, I used an embedding layer and an LSTM layer • For the decoder, I used another LSTM layer followed by a dense layer

Bidaf implemented on SQUAD 1.1 dataset to solve Question Answering problem statement: https://github.com/rajpurkar/SQuAD-explorer/tree/master/dataset Implemented Bi-directional attention flow(BiDAF) network from scratch as proposed in Minjoon Seo et. Al on Stanford Question Answering Dataset v1.1 to build a closed-domain, extractive Q&A model which can extract a span of text from the context as the answer.

Implementation of Pretrained models: Implementation of pre trained models like BERT, SCIBERT, CLINICAL BERT, XLNET to generate embedding and extract information from twitter data, using text extraction method powered by pretrained models. 

Embedding and Knowledge Graph :

Projects involving solving problems using embedding generation algorithms like DEEPWALK, NODE2VEC, LINE from scratch. Further these embedding were evaluated using basic methods like Jaccard coefficient score and Resource Allocation score. More evalution was performed using much sophisticated methods like LINK PREDICTION, EDGE CLASSIFICATION etc. 
THE DETAIL FINDINGS AND RESULTS have been documented and could be found within the folder. 

Recommendation System: 

Creating dynamic Recommedation system to solve cold start problem using Joint Emedding and dual RNN. This algorithm leverage the user product interaction to simultaneously update both the product as well as user embedding. 

Speech: 

Contains code for speech denoising architecture along with documentation for architecture that was proposed in FLIPKART GRID 2.0 software development challenge. 
The details is incorporated in the pdf attached. 

Computer Vision: 

Mnsit_CNN.ipynb : Dataset: http://yann.lecun.com/exdb/mnist/ The MNIST problem is a dataset developed by Yann LeCun, Corinna Cortes and Christopher Burges for evaluating machine learning models on the handwritten digit classification problem. The dataset was constructed from a number of scanned document dataset available from the National Institute of Standards and Technology (NIST). This is where the name for the dataset comes from, as the Modified NIST or MNIST dataset. Each image is a 28 by 28 pixel square (784 pixels total). A standard split of the dataset is used to evaluate and compare models, where 60,000 images are used to train a model and a separate set of 10,000 images are used to test it. It is a digit recognition task. As such there are 10 digits (0 to 9) or 10 classes to predict. Results are reported using prediction error. Convolutional Neural Networks has been used in the model to obtain prediction error of less than 1%.

cifar_10_imageclassification.ipynb: Dataset: https://www.cs.toronto.edu/~kriz/cifar.html CIFAR-10 (Canadian Institute For Advanced Research) is the “hello world” type dataset of computer vision. Used deep leaning Convolutional neural network to build the model using Keras API for Image Classification with 88.6% accuracy on test set.

GAN(Generative Adversarial networks) Project

Monet2photo_cyclegan.ipynb : Dataset: https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip Studied the construct and the underlying architecture of simple GAN and CycleGAN. Implemented CycleGAN model for Painting style neural transfer using ‘monet2photo’ dataset which generated images of photos from images of Monet paintings, and vice-versa in absence of one-to-one correspondence between input and output images.

horse2zebra_cyclegan.ipynb: Dataset: https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip Applied CycleGAN model on the Horse Zebra dataset used by Zhu et al. (Research Paper: https://arxiv.org/abs/1703.10593) in keras.

