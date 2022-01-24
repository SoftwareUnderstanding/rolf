# Name under curent development

## Overview
The goal of our project is to do face and facial landmarks detection, highlighting important points of the face in order offer support for emotion recognition for persons who are not able to do so naturally (Asperger's syndrome).
The classification algorithm can be run on either a photograph of a person, loaded into the GUI, or live on video.

For the GUI, we used the Python module called TKinter: https://docs.python.org/2/library/tkinter.html

For the Neural Network we used a detector using the dlib library.
 
The dataset which we trained the NN on is the iBUG 300-W dataset: https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/

## AI
* everything is developed using the PyTorch Python module: https://pytorch.org/
* we used the Squeezenet architecture proposed by Iandola et. al (https://arxiv.org/abs/1602.07360), pretrained on the Imagenet dataset;
* trained it using the RAdam optimizer proposed by Liu et. al (https://arxiv.org/abs/1908.03265), with a learning rate of 1e-5, batch size of 8 samples, 0.001 learning rate decay, for 100 epochs;
* Results: 89% train accuracy, 39% test accuracy. Conclusions: overfitting hard, for now;
* dataset: http://app.visgraf.impa.br/database/faces/
