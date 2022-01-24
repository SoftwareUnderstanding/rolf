# ActivityRecognitionKeras

This repository performs a simple experiment on the UCI Human Activity Recognition dataset, available <a href="https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones">here</a> and described in the following paper:

<blockquote cite = "https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones">
Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013. 
</blockquote>

We aim to observe the model's performance under 3 experimental conditions: 1) The model is trained only on accelerometer features 2) the model is trained only on gyroscope features 3) the model is trained on both accelerometer and gyroscope features.

The convolutional architecture is an implementation very close to that used in the below:

<blockquote cite="https://www.sciencedirect.com/science/article/pii/S0957417416302056">
Ronao, Charissa Ann, and Sung-Bae Cho. "Human activity recognition with smartphone sensors using deep learning neural networks." Expert systems with applications 59 (2016): 235-244.
</blockquote>

We train the models using the 1cycle policy as described by fast.ai <a href="https://sgugger.github.io/the-1cycle-policy.html#the-1cycle-policy">here</a>, based upon

<blockquote cite="https://arxiv.org/abs/1803.09820">
Smith, Leslie N. "A disciplined approach to neural network hyper-parameters: Part 1--learning rate, batch size, momentum, and weight decay." arXiv preprint arXiv:1803.09820 (2018).
</blockquote>

Results are reported with accuracy, precision and recall by class, and confusion matrices.

-----

# To run

Requirements are effectively the same as those to run the base docker image ```tensorflow/tensorflow:1.14.0-gpu-py3```. For GPU computation, requires cuda > 10.0. If this requirement is met, then use:

```
docker build --tag matthewalmeida/activityrecognitionkeras .
```

to build the image, and run with:

```
docker run --rm --runtime=nvidia matthewalmeida/activityrecognitionkeras
```

To run on CPU, simply remove ```--runtime=nvidia``` from the docker run command.
