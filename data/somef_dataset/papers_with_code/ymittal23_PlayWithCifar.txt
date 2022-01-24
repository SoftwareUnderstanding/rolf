# PlayWithCifar
Objective: Image Classification on Cifar 10 dataset


Requirements: Since my google credits are expired I have decided to build this project on Google Colab. The project will use keras with tensorflow as backend. We don't need to install libraries on Colab and can start straight away. 

Use below link to open the notebook and run the code:

https://colab.research.google.com/drive/1654IUaU8VR2mFcemMC_E5E_3Bp6JY2lq


#### Model:

Since I am using Google Colab it will be little difficult to train heavy models like WideResnet which in turn can give really good accuracy (ex- https://github.com/fastai/fastai/blob/master/examples/cifar.ipynb) and tune them in a single day (I planned to finish this in a day), so I have decided to go with not so famous but light architecture 'DavidNet'. There is an online competition about fast training called DAWNBench, and the winner was David C. Page in April 2019, who built a custom 9-layer Residual ConvNet, or ResNet. Here I have used a 3 layer simple architecture referred as 'DavidNet'.


Below is the complete architecture. Download it for better understanding. Picture is self explantory.


![DavidNet](https://github.com/ymittal23/PlayWithCifar/blob/master/davidnet.png)



In DavidNet, training images go through the standard Cifar10 transformations, that is, pad 4 pixels to 40x40, crop back to 32x32, and randomly flip left and right. In addition, it applies the popular Cutout augmentation (https://arxiv.org/pdf/1708.04552.pdf) as a regularization measure, which alleviates overfitting. DavidNet trains the model with Stochastic Gradient Descent with Nesterov momentum (https://dominikschmidt.xyz/nesterov-momentum/), with a slanted triangular learning rate schedule.

We are using the same hyperparameters used in original model made by David here. Since it is a classification problem I am using categorical cross entropy as my loss function. It can be calculated as −(ylog(p)+(1−y)log(1−p)). 
One cycle policy is used for picking the right learning rate for training model (https://arxiv.org/pdf/1803.09820.pdf).

91.1 % accuracy is achieved with this light and simple model. 

#### Problems Faced:
Making it work in limited time was not easy. Challenge was to achieve more than 90% accuracy with limited resources and time.

#### What could be improved?
Given enough time more data augmentation could help this model. Adding more resnet layers(making model deeper) could also help us achieve better accuracy but this will also increse training time of the model. 



