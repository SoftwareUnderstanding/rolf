# Emotion Recognition

Nobel laureate Herbert Simon wrote, "In order to have anything like a complete theory of human rationality, we have to understand what role emotion plays in it."

Industries are rapidly changing with the rapid growth of Artifical Intelligence. When it comes to understand human decisions, there are various factors that we have to take into consideration and one of them is emotion. Emotions can act as a bias to many of our day to day decision making. In this project, I attempt to use the 'Chicago Faces Database' to identify one of 4 emotions: Happy, Fear, Anger, and Neutral.

The 'Chicago Faces Database' is unbalanced. The approximate distribution of models is shown below:
```
Neutral - 50%
Happy - 25%
Fear - 12.5%
Anger - 12.5%
```

As a solution to this distribution, I trained to models which are the 'NeutralModel' and the 'EmotionModel'. The NeutralModel predicts whether the emotion shown is Neutral or not. If the model proves negative in the NeutralModel, the image is sent to the EmotionModel which predicts the specific emotion. An attempt of dividing the EmotionModel into a 'HappyModel' and a 'FearOrAngerModel' will be tested to check if it performs better. The current model acts as a 'binary -> categorical' sequence. Dividing the EmotionModel will be like having a DecisionTree.

The model is trained using TensorFlow-GPU. The validation set is 20% of the original data. The NeutralModel performed with a peak validation accuracy of 79%. The EmotionModel performed with a peak validation accuracy of 84%. 


Demo below shows the sample performance of the the past model.
![](src/demo.gif)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Make sure you have the following installed (or the latest version):

```
Python 3.6
Numpy 1.15.4
Pandas 0.23.4
OpenCV2 3.4.4
Tensorflow 1.12.0
```

### Installing

Simply fork notebook into your local directory.

## Deployment

Assuming you have all necessary modules installed. Through your command prompt, move to the local repository and run the command:

```
python GrabScreen.py
```
A window would open mirroring a portion of your screen. Simply move the image over that portion of the screen and the predicted emotion is shown on the upper left.
For best performance, let the face occupy the entire window.

## Authors

* **Prince Mallari** - (https://github.com/pmallari)

## Acknowledgments

* Prudhvi Raj Dachapally (https://arxiv.org/ftp/arxiv/papers/1706/1706.01509.pdf)

* University of Chicago Center for Decision Research (https://chicagofaces.org/)

* Ian J. Goodfellow (https://arxiv.org/pdf/1302.4389v3.pdf) & (https://www.kaggle.com/c/facial-keypoints-detector)
* David Warde-Farley
* Mehdi Mirza
* Aaron Courville
* Yoshua Bengio

* Harrison Kinsley (https://www.youtube.com/user/sentdex)
