## Using elu on Keras to achieve 95% accuracy on Cifar-10 dataset
elu = Exponential Linear Units
Referencing the article in https://arxiv.org/abs/1511.07289v1
ELU "speeds up learning in deep neural networks and leads to higher classification accuracies" and also "ELUs lead not only to faster learning, but also to better generalization performance once networks have many layers (>= 5"
Using the model published by the keras team: https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py as the base model (w/o image augmentation) for comparison.

I was able to get 75% accuracy with 20 epochs with the base model.

    Train on 50000 samples, validate on 10000 samples
Epoch 1/20
50000/50000 [=====================] - 59s 1ms/step - loss: 1.8266 - acc: 0.3307 - val_loss: 1.5874 - val_acc: 0.4371
Epoch 2/20
50000/50000 [=====================] - 42s 845us/step - loss: 1.5042 - acc: 0.4534 - val_loss: 1.4038 - val_acc: 0.5006
:
:
:
Epoch 18/20
50000/50000 [=====================] - 41s 825us/step - loss: 0.7802 - acc: 0.7313 - val_loss: 0.7838 - val_acc: 0.7340
Epoch 19/20
50000/50000 [=====================] - 41s 813us/step - loss: 0.7692 - acc: 0.7366 - val_loss: 0.7380 - val_acc: 0.7462
Epoch 20/20
50000/50000 [=====================] - 41s 814us/step - loss: 0.7545 - acc: 0.7409 - val_loss: 0.7362 - val_acc: 0.7489

### ELU is awesome!
Switching to elu, it returned 95% accuracy! And also faster...

    Train on 50000 samples, validate on 10000 samples
Epoch 1/20
50000/50000 [===================] - 39s 782us/step - loss: 1.5519 - acc: 0.4505 - val_loss: 1.3890 - val_acc: 0.5069
Epoch 2/20
50000/50000 [==================] - 39s 789us/step - loss: 1.2326 - acc: 0.5713 - val_loss: 1.1883 - val_acc: 0.5878
Epoch 3/20
50000/50000 [===================] - 39s 776us/step - loss: 1.0905 - acc: 0.6217 - val_loss: 1.1572 - val_acc: 0.6045
:
:
:
Epoch 18/20
50000/50000 [===================] - 39s 774us/step - loss: 0.2033 - acc: 0.9349 - val_loss: 1.0350 - val_acc: 0.7375
Epoch 19/20
50000/50000 [===================] - 39s 775us/step - loss: 0.1642 - acc: 0.9489 - val_loss: 1.1101 - val_acc: 0.7324
Epoch 20/20
50000/50000 [===================] - 39s 779us/step - loss: 0.1296 - acc: 0.9598 - val_loss: 1.1817 - val_acc: 0.7277

### Training Accuracy Up, Validation Accuracy Down
The strange thing however, is that although the accuracy went up, the validation accuracy was not better.

The python file contains both models.  Be sure to comment out one of the model before running.
