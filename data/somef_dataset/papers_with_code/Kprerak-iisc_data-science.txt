# data-science
Data science Projects 

NLP: Contains projects related to Natural Lnaguage Processing

BiDAF_QA_SQuAD.ipynb : Dataset: https://github.com/rajpurkar/SQuAD-explorer/tree/master/dataset
Implemented Bi-directional attention flow(BiDAF) network from scratch as proposed in Minjoon Seo et. Al on Stanford Question Answering Dataset v1.1 to build a closed-domain, extractive Q&A model which can extract a span of text from the context as the answer.

Nmt_seq2seq: Neural machine translation from English to German and again back to English from German.
dataset: http://www.manythings.org/anki/deu-eng.zip 
• Implemented Encoder- decoder based Sequence-to-Sequence (seq2seq) model on only 20k data out of over 150,000 data due to less computational power. 
• For the encoder, I used an embedding layer and an LSTM layer
• For the decoder, I used another LSTM layer followed by a dense layer

Topic_modeling_consumers:  Discovering Hidden Semantic Structures of Texts from Large Corpus of Documents
Dataset: https://drive.google.com/file/d/141NT1NvZGaBPPzdXStsATJzLZYa77uTw/view?usp=sharing
• Developed a probabilistic model to cluster the abstract topics present in Consumer complaints in Financial domain from CFPB website.
• Used Reg-ex based text regularization, Stop words removal, Stemming for pre-processing and Tfidf Vectorizer to build featurevector.
• Implemented Topic Modeling using Latent Dirichlet Allocation (LDA) to classify each consumer complaints into one of the six topics.

Topic Modeling using Latent Dirichlet Allocation (LDA) on https://www.kaggle.com/benhamner/nips-papers dataset.


ML: Projects related to Machine Learning

cred_bal.ipynb: Dataset: https://rdrr.io/cran/ISLR/man/Credit.html
•	Objective: Analyze the dataset consisting of information from credit card holders to comprehend which factors influence the Credit Card Balance of a cardholder, and to predict the average Balance of a given individual. 
•	Carried out EDA of the dataset, followed by feature selection and regression analysis to build a model which explains 96% variance. 

fraud.ipynb: Dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud
•	Developed various predictive models to compare their efficiency in detecting fraudulent transactions based on their F1 score.
•	Applied SMOTE for oversampling the minority class and used Logistic regression, K-nearest neighbour, support vector classifier and decision tree classifier.
•	Used Principal component analysis(PCA) and t-SNE for clustering fraudulent transactions and non-fraudulent transactions separately.
•	Best result: XGBoost classifier was able to detect more than 80% fraud transactions without classifying a lot of non-frauds as fraudulent.


Vision: Projects related to Computer Vision

Amazon.ipynb :  Dataset : https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data
• To label satellite images provided by Planet and SCCON with atmospheric conditions and various classes of land cover/use. 
• Developed a convolutional neural network for multi-label photo classification to achieve F1 score of 0.837 using only 10% of data.
• Used Transfer learning to improve the model performance using VGG-16 model to achieve F1 score of 0.886 with only 10% of data.

Mnsit_CNN.ipynb : Dataset: http://yann.lecun.com/exdb/mnist/
The MNIST problem is a dataset developed by Yann LeCun, Corinna Cortes and Christopher Burges for evaluating machine learning models on the handwritten digit classification problem. The dataset was constructed from a number of scanned document dataset available from the National Institute of Standards and Technology (NIST). This is where the name for the dataset comes from, as the Modified NIST or MNIST dataset. Each image is a 28 by 28 pixel square (784 pixels total). A standard split of the dataset is used to evaluate and compare models, where 60,000 images are used to train a model and a separate set of 10,000 images are used to test it.
It is a digit recognition task. As such there are 10 digits (0 to 9) or 10 classes to predict. Results are reported using prediction error. Convolutional Neural Networks has been used in the model to obtain prediction error of less than 1%.

cifar_10_imageclassification.ipynb: Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
CIFAR-10 (Canadian Institute For Advanced Research) is the “hello world” type dataset of computer vision. Used deep leaning Convolutional neural network to build the model using Keras API for Image Classification with 88.6% accuracy on test set.

GAN(Generative Adversarial networks) Project

Monet2photo_cyclegan.ipynb : Dataset: https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip
Studied the construct and the underlying architecture of simple GAN and CycleGAN. Implemented CycleGAN model for Painting style neural transfer using ‘monet2photo’ dataset which generated images of photos from images of Monet paintings, and vice-versa in absence of one-to-one correspondence between input and output images. 

horse2zebra_cyclegan.ipynb:  Dataset: https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip
Applied CycleGAN model on the Horse Zebra   dataset used by Zhu et al. (Research Paper: https://arxiv.org/abs/1703.10593) in keras
 


Time Series: Project related to Time Series Analysis And Forecasting

trade.r : dataset: http://mgmt.iisc.ac.in/CM/MG222/Data_Files/gdp.data
•	Built an appropriate SARIMA model to model the trend and seasonality present in the time series data of Quarterly GDP of India from 1996-97:Q1 to 2013-14:Q2. Validated if the residuals of the model satisfied all the assumptions using graphical and statistical methods
•	Forecasted the GDP next 4 Quarters using the fitted model with Mean Absolute Percentage Error(MAPE) equal to 2.18%




