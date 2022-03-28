# ml-examples
These are Machine Learning examples I have created or adapted to give you a better understanding of my machine learning understanding. - Hamilton

### CNN SVHN Classification
* I created a CNN (Convolutional Neural Network) model to classify Street View House Numbers (SVHN)
* This was the capstone project for the Coursera class Getting Started with Tensorflow 2 class 
* Current students of the Coursera Getting tarted with Tensorflow 2 class should not look at this example 
* [CNN_SVHN_TF2_Capstone_Project_by_Hamilton_2020_12_3.pdf](CNN_SVHN_TF2_Capstone_Project_by_Hamilton_2020_12_3.pdf) \- [.ipynb](CNN_SVHN_TF2_Capstone_Project_by_Hamilton_2020_12_3.ipynb)

### Language Translation Model using Encoder RNN and Decoder RNN
* I created this model that translated from English to Germman
* This was the capstone project for the Coursera Customzing Your Models with Tensorflow 2 class 
* This project taught us Encoder/Decoder seq2seq architectures, using LSTMs (Long Short Term Memory)
* This project was for learning purposes only and not production
* Current students of this Customizing Your Models with Tensorflow 2 class should not look at this
* [Neural_Translation_Model_Capstone_Project_by_Hamilton_2021_1_12.pdf](Neural_Translation_Model_Capstone_Project_by_Hamilton_2021_1_12.pdf) \- [.ipynb](Neural_Translation_Model_Capstone_Project_by_Hamilton_2021_1_12.ipynb)

### Transformer Model Implementation for Language Translation
* Translates Russian to English 
* Implements a Transformer model including attention in Keras / Tensorflow along with the BERT subword tokenizer
* Transformers are the state of the art for Natural Language Processing in Machine Learning
* [Russian_Transformer_Model_for_Language_Translation_v2.pdf](Russian_Transformer_Model_for_Language_Translation_v2.pdf) \- [.ipynb](Russian_Transformer_Model_for_Language_Translation_v2.ipynb)

### T5 Model and HuggingFace Framework Language Translation
* Translates the same 5 English strings (all accurately) as my Capstone project above from English to German
* Translates using the HuggingFace pipeline, and with slightly lower level calls to the T5 and MarianMT models
* [Language_Translation_Using_the_T5_Model_And_HuggingFace_Framework.ipynb](Language_Translation_Using_the_T5_Model_And_HuggingFace_Framework.ipynb) 

### Sentiment Analysis - Fine Tuning BERT model on IMDB
* This example downloads BERT (Bidirectional Encoder Representations from Transformers) model from tfdev.hub
* It also serves the model for inference via TensorFlow Serving
* Trained the BERT model on the IMDB Movie Review dataset to make positive and negative sentiment classification predictions
* [Sentiment_Analysis_Fine_Tuning_a_BERT_model_on_IMDB.ipynb](Sentiment_Analysis_Fine_Tuning_a_BERT_model_on_IMDB.ipynb)

### Question Answering using BERT, Roberta & Electra Models Pretrained on Squad & Squad 2
* I adapted these short examples using the HuggingFace API
* [Question_Answering_Models_BERT_Roberta_Electra_Pretrained_on_Squad2.ipynb](Question_Answering_Models_BERT_Roberta_Electra_Pretrained_on_Squad2.ipynb)

### BERT GLUE E2E on TPU Notebook
* This example downloads BERT (Bidirectional Encoder Representations from Transformers) model from tfdev.hub
* It fine tunes the training on one (any one) of the GLUE (General Language Understanding Evaluation) datasets
* More info about GLUE datasets can be found at: https://arxiv.org/pdf/1909.13719.pdf
* This example is copied from: https://www.tensorflow.org/tutorials/text/solve_glue_tasks_using_bert_on_tpu
* [BERT_Glue_E2E.ipynb](BERT_Glue_E2E.ipynb)

### Image Segmentation Using U-Net
* Segments dogs and cats out of images taken from the Oxford IIT Pet Dataset
* [Image_Segmentation_Using_U-Net.pdf](Image_Segmentation_Using_U-Net.pdf) - [.ipynb](Image_Segmentation_Using_U-Net.ipynb)

### Image Object Detection and Instance Segmentation 
* Compares Mask R-CNN using ResNet V2 and EfficientDet D7 for Object Detection for Object Detection
* Uses Tensorflow 2 with TFHub
* [Object_Detection_Inference_Using_TF2_and_TFHub.pdf](Object_Detection_Inference_Using_TF2_and_TFHub.pdf) - [.ipynb](Object_Detection_Inference_Using_TF2_and_TFHub.ipynb)

### Video Instance Segmentation and Object Detection Example 
* Created this [Mask R-CNN Instance Segmentation and Object Detection Video](https://drive.google.com/file/d/1M9OqeQGM_KzHUGdlOYmcDup-jh0N-Zms/view?usp=sharing)

### Live Demo of Blender Chatbot using the RoBERTa Transformer
* I setup this demo with a modestly improved UI on Google Cloud using Docker
* See a Blenderbot [example conversation](sample-blender-conversation.png)
* Try it out at: https://blenderbot90m-wg5fqcbcta-uw.a.run.app (Can take 30 seconds for Google Cloud to load docker image in and startup) 
* Note, this demo uses the 90 million parameter small model. The much larger 2.7 billion and 9.4 billion parameters models will produce better conversations and can be run on more expensive hardware.

* Here's the paper on BlenderBot developed by the Facebook AI team: https://arxiv.org/pdf/2004.13637.pdf* For comparison here's Mitsuki bot from Pandorabots which I did not find as good: https://https://chat.kuki.ai/
