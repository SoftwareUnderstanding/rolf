# nlp-classification-with-BERT
view the jupyter notebook:
[bert-tutorial](https://nbviewer.jupyter.org/github/surancy/nlp-classification-with-BERT/blob/master/bert_tutorial.ipynb)

This tutorial introduces how to implement the state-of-art NLP language model, BERT, using the Huggingface Transformer library. This tutorial will walk you through the introduction of BERT, overview of some NLP tasks, specifically GLUE dataset that is used for sentence understanding, followed by the introduction of 🤗Transformer, code examples of training BERT with GLUE dataset built in Tensorflow, and using the pre-trained BERT model to predict some new instances.


```markdown
Model performance report for 2 epochs at step size = 115

Train for 115 steps, validate for 7 steps
Epoch 1/2
115/115 \[==============================\] - 1515s 13s/step - loss: 0.5822 - accuracy: 0.6933 - val_loss: 0.4512 - val_accuracy: 0.7917
Epoch 2/2
115/115 \[==============================\] - 1499s 13s/step - loss: 0.3312 - accuracy: 0.8550 - val_loss: 0.4058 - val_accuracy: 0.8309

Model performance report for 2 epochs at step size = 115

Train for 25 steps, validate for 6 steps
Epoch 1/6
25/25 [==============================] - 389s 16s/step - loss: 0.4844 - accuracy: 0.8037 - val_loss: 0.5792 - val_accuracy: 0.7318
Epoch 2/6
25/25 [==============================] - 394s 16s/step - loss: 0.5309 - accuracy: 0.7475 - val_loss: 0.5240 - val_accuracy: 0.7578
Epoch 3/6
25/25 [==============================] - 375s 15s/step - loss: 0.5382 - accuracy: 0.7425 - val_loss: 0.4910 - val_accuracy: 0.7839
Epoch 4/6
25/25 [==============================] - 377s 15s/step - loss: 0.4638 - accuracy: 0.7975 - val_loss: 0.4382 - val_accuracy: 0.8047
Epoch 5/6
25/25 [==============================] - 384s 15s/step - loss: 0.3969 - accuracy: 0.8147 - val_loss: 0.4756 - val_accuracy: 0.7995
Epoch 6/6
25/25 [==============================] - 391s 16s/step - loss: 0.2941 - accuracy: 0.8875 - val_loss: 0.4644 - val_accuracy: 0.7969
```
![bertperformance](bertperformance.png)

# Citations
1. Google Research: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. October 2018. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, from: https://arxiv.org/abs/1810.04805
2. BERT Open-source: https://github.com/google-research/bert
3. Huggingface Transformers: https://github.com/huggingface/transformers
4. Wilson L Taylor. 1953. Cloze procedure: A new tool for measuring readability. Journalism Bulletin, 30(4):415–433.
5. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Pro- cessing Systems, pages 6000–6010.
6. Tensorflow: https://www.tensorflow.org/
7. PyTorch: https://pytorch.org/
8. Tensorflow dataset: https://medium.com/tensorflow/introducing-tensorflow-datasets-c7f01f7e19f3
9. GLUE/MRPC dataset: https://www.microsoft.com/en-us/download/details.aspx?id=52398
