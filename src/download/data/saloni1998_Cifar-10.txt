# Cifar-10
Attached notebooks are implementation of ResNet proposed by Microsoft Research in
https://arxiv.org/abs/1512.03385 and self-proposed model.<br>
<b>Summary of Contents:</b>
1) <b>cifar_10.ipynb</b>: A 34 layer Deep Convolutional Neural Network based on Resnet
Architecture.
2) <b>cifar_110_layers.ipynb</b>: 110 layer implementation of Deep Convolutional
Neural Network based on Resnet Architecture.
3) <b>cifar_10_87.ipynb</b>: self proposed model
<br>
<b> Dataset used :</b><br>
<b>Link of dataset</b>: https://www.cs.toronto.edu/~kriz/cifar.html

The deep residual network proposed in the paper was tested on various datasets like
MS-COCO,Cifar-10 etc. Keeping in view the time and computational requirements I
have used Cifar-10 dataset composed of 60,000 images (50,000 training & 10,000
testing) divided into 10 classes.
The models are trained on GoogleColab and using the same hyperparameters and
optimizer as given in the paper.
Data augmentation methods like horizontal flipping has been used as discussed in the
paper.

<b>Results</b><br>
Using my proposed architecture inspired from the stacking concept of vgg , highest validation accuracy achieved is <b>87.68%</b> <br>
Using Cifar 34 layer architecture, highest validation accuracy achieved is 85.86%.<br>
Using Cifar 110 layer architecture, highest validation accuracy achieved is 70.02%.<br>
