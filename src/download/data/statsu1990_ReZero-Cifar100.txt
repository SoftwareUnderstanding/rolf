# ReZero-Cifar100
Unofficial verification of ReZero ResNet on cifar 100 dataset<br>


## ReZero
ReZero is All You Need: Fast Convergence at Large Depth<br>
https://arxiv.org/abs/2003.04887<br>

Unofficial pytorch implementation of ReZero in ResNet<br>
https://github.com/fabio-deep/ReZero-ResNet<br>

## Verification
Trained PreAct-ResNet with Cifar100 and verified how accuracy and convergence change with and without ReZero.<br>

#### Condition
- Data <br>
  - cifar 100 <br>
- Model <br>
  - Base model: PreAct ResNet 18, 50 <br>
  [https://arxiv.org/abs/1603.05027:title] <br>
  - Model with ReZero: All the residual connections in the base model are changed to ReZero connections. The initial value of residual weight α is 0. <br>
  - Model with ReZero (personally improved version) : All the residual connections in the base model have been changed to ReZero connections. Use tanh (α) instead of α (initial value of α is 0). To prevent α from becoming abnormally large, we used tanh (α) for the purpose of limiting the value range. <br>
- Learning method <br>
  - Cross entropy loss <br>
  - SGD, learning rate 0.1 (reduce learning rate by 0.2 for 60, 120, 160 epoch), 200 epochs, batch size 128 <br>
  - Data augmentation (random flip, random shift scale rorate) <br>

#### Result
The accuracy and convergence did not improve.<br>

###### PreAct ResNet 18
![mrc](https://github.com/statsu1990/ReZero-Cifar100/blob/master/results/loss_preact-resnet18.jpg)<br>
![mrc](https://github.com/statsu1990/ReZero-Cifar100/blob/master/results/accuracy_preact-resnet18.jpg)<br>
![mrc](https://github.com/statsu1990/ReZero-Cifar100/blob/master/results/alpha_preact-resnet18.jpg)<br>

###### PreAct ResNet 50
![mrc](https://github.com/statsu1990/ReZero-Cifar100/blob/master/results/loss_preact-resnet50.jpg)<br>
![mrc](https://github.com/statsu1990/ReZero-Cifar100/blob/master/results/accuracy_preact-resnet50.jpg)<br>
![mrc](https://github.com/statsu1990/ReZero-Cifar100/blob/master/results/alpha_preact-resnet50.jpg)<br>
