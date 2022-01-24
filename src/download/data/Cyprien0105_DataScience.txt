# DataScience

The purpose of this little project is to provide simple tools to use transfer learning models in a very simple way.
The first model available is the efficientNetB3 trained on the imagenet dataset. The goal is to classsify images by using a powerfull neuralnetwork.


imagenet dataset : http://www.image-net.org/

efficientnet neural network : https://arxiv.org/abs/1905.11946


To use and retrain the model, you need to provide your own labeled image dataset.
You have to store your images in folders : one folder = one class. So if you want to apply the model on 3 classes, you have to create 3 folders, sort your images by classes and store each class in one folder. Give the same name to the folder as the class it contains.


Here an example of dataset used in the example folder : https://intra.ece.ucr.edu/~mbappy/pubs/README.pdf

It contains real estates images divided in six classes.
