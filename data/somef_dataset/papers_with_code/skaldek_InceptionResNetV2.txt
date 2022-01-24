# Inception-ResNet-V2
It is my try of tensorflow implementation. Everything was read here:
https://arxiv.org/pdf/1602.07261.pdf
![Model](/fig.png/)

### Network Training
In order to start training you need to follow a few simple steps. 
Images for classification should be divided into folders, each of which will contain a separate class and placed in the Images folder, 
which will be located in the root. For example, you can [use my fork](https://github.com/skaldek/ImageNet-Datasets-Downloader).
Then it remains to run only the train.py file. In addition, 
it is possible to edit some parameters in the hyperparams.py file, it may be worth reducing dropout. 
Checkpoints will be saved every epoch.

### Prediction
Later, it is possible to predict belonging to a certain class of photos using predict.py, 
only you will need to change the path to the file in it.

### Differences from the original paper
The number of filters in some convolutional layers may differ from the arxiv, 
this is due to mishmash in it.
