# CIFAR-10
## Image Classification on CIFAR-10 Dataset | Test Data Accuracy: 96.82%


In this implementation, I have used the WideResNet Architecture [[1]](https://arxiv.org/abs/1605.07146) to increase performance, I've adapted xternalz's efficent implementation [[2]](https://github.com/xternalz/WideResNet-pytorch
). 
The Image dataset is normalized by per channel mean and standard deviation. 


### For Data Augementation following technquies are used:
>- Random Cropping
>- Random Horizontal flips
>- Cutout Regularization

### Training:
The WideResNet is trained on the following settings:
` depth = 28, 
widen_factor = 10`


Along with the afore-mentioned 'cutout', a drop-out rate of 0.3 is used to avoid over-fitting.

The WiderResNet CNN model is trained on the cutout dataset for `180 epochs` with an `initial learning rate of 0.1`. 
The learning rate is step decreased at `40, 60, 80, 90, 150, 155 by a factor of 5x (*0.2)`

A Stochastic Gradient Descent optimizer with `momentum 0.9`, a `weight decay of 5e-4`, `Nesterov momentum set to True` 

A loss criterion of cross-entropy is used. 

The Data augmentation and regularization benefits are provided by cutout [[3]](https://arxiv.org/pdf/1708.04552.pdf). Cutout Regularization works by cuting out random grid or square holes from the train images, promoting the model to learn upon the finer details in the images.
>This shows an interseting observation of test accuracy being greater than train accuracy during the course of training. 

The cutout regularization adds two new hyper-parameters:
> - num_holes : the number of cutout holes to create for augmentation
> - length : the length of the holes in dimensions

### Testing:

A model is tested on the 10,000 samples of test_batch after per-channel normalization. 

The highest accuracy achieved by the model was ***0.9682***
with, `num_holes = 1 and length = 16`
after which no amount of hyper-parameter tuning broke this performance ceiling.

with, `num_holes = 2` the model hits a ceiling at *0.956* 

Included py script test.py can be used to test .pt (torch checkpoint) for testing accuracy scores.

> ##### NOTE:
> - To see training logs, prefer the logs.csv as google colab turncates cell output after a certain buffer limit.
> - Change the `pt_file_path` variable in test.py to test any pytorch checkpoint.


###### References:
- [1] https://arxiv.org/abs/1605.07146
- [2] https://github.com/xternalz/WideResNet-pytorch
- [3] https://arxiv.org/pdf/1708.04552.pdf


