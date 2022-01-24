### Image Recognition on Google Street View House Numbers (SVHN) Dataset using Wide Residual Networks (WRNs)

Problem statement:
To correctly classify the numbers in SVHN (32x32) Dataset.

Dataset source: http://ufldl.stanford.edu/housenumbers/ - Format 2, Cropped Digits

Project Summary:
Given that SVHN is a published dataset, many researchers would use it to benchmark their model. One model which outperforms the others in terms of efficiency, accuracy and code availability is the WideResNet. It is built on residual learning, with a small twist on the architecture, focusing on width rather than depth. By avoiding the problem of vanishing gradients in Convolutional Neural Networks and reducing complexities by increasing width, it is the most efficient and accurate model among the 6 models that we have built. This provided a baseline accuracy of 93%.
Upon deciding on the model, we decided to preprocess the data. As the number 0 was assigned to the label 10 instead of label 0, one-hot encoding introduced an empty column which we removed, reducing the runtime. Recalling from the workshop, image standardisation was implemented by subtracting the mean and dividing by the standard deviation. Additionally, we decided to include the extra dataset in SVHN. However, together with data augmentation, it proved to be too computationally expensive and thus we removed it and half of the extra dataset. These changes increased the accuracy to 95%.
Next, although we initially decided to use the Adam optimiser due to its quick convergence, we ultimately decided on the SGD optimiser with momentum as it can converge to a better local minimum. To prevent overfitting, we implemented ReduceLROnPlateau and EarlyStopping, which decreases the learn rate when the val_loss does not improve, and stops training the model if necessary. To capture the best model weights, ModelCheckpoint is used, and weights are loaded when ReduceLROnPlateau is called, reducing the runtime by 6 epochs.
Overall, many of the methods we tried produced undesirable results. However, with grit and determination, we built on those that were promising, achieving a personal best of 96.8%. Going further, we are certain that the model can still be improved, given greater computational power to utilise the remaining half of the extra dataset, and by introducing data augmentation.

Project Presentation: https://sway.office.com/QNYrIkASCjPH90Ib - Has graphs explaining hyperparameters optimisation

Final Accuracy: 96.82698217578365 (load model_weights.h5), 45mins to train<br>
Limitations: Computing power<br>
To further increase the accuracy to reach >98%, as done in the research paper, it is preferable to increase the width of the model to k=8-10, utilise the whole extra dataset images and implement a dropout of 0.4. However, using the free google colaboratory, some compromises on the accuracy had to be made.

Credits:<br>
https://arxiv.org/pdf/1605.07146v1.pdf - Research paper on WRN<br>
https://github.com/titu1994/Wide-Residual-Networks/blob/master/wide_residual_network.py - the keras implementation of WRN<br>
https://github.com/meliketoy/wide-residual-network - implementation details
<br>
<br>
Good Reads:

How to save and load model weights (Callback)<br>
https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/save_and_restore_models.ipynb#scrollTo=xCUREq7WXgvg

Who is best at SVHN<br>
https://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#5356484e<br>
https://benchmarks.ai/svhn<br>
https://github.com/MatthieuCourbariaux/BinaryConnect - Theano<br>
https://github.com/Coderx7/SimpleNet - Caffe and Pytorch

Preprocessing hacks<br>
http://ufldl.stanford.edu/wiki/index.php/Data_Preprocessing

How to boost your accuracy<br>
https://machinelearningmastery.com/improve-deep-learning-performance/
