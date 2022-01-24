## ResNet Implementation
This repository includes ResNet18, ResNet34, ResNet50, ResNet101, ResNet152 in Tensorflow 2.  
I used tf.keras.Model and tf.layers.Layer instead of tf.keras.models.Sequential.  
This allows us to customize and have full control of the model.  
Also, I used custom training instead of relying on the fit() function.
In case we have very huge dataset, I applied online loading (by batch) instead of loading the data completely at the beginning. This will eventually not consume the memory.  

####  Architectures for ResNet
<p></p>
<center>  
<img src="img/2.png" align="center" width="500" height="1000"/>
<p></p>
Figure 1. Example network architectures for ImageNet. Left: the
VGG-19 model . Middle: a plain network with 34 parameter layers .
Right: a residual network with 34 parameter layers.
</center>  
<p></p>
<center>   
<img src="img/3.png" width="800" height="400"/>   
<p></p>
Table 1. Architectures for ImageNet. Building blocks are shown in brackets (see also Fig. 5), with the numbers of blocks stacked. Downsampling is performed by conv3 1, conv4 1, and conv5 1 with a stride of 2.   
</center>

#### The Block Architecture
<center>  
<img src="img/111.png" width="500" height="250"/>   
<p></p>
Images are taken from [source](https://arxiv.org/pdf/1512.03385.pdf)   
</center>

#### Training on MNIST
<p></p>
<center>  
<img src="img/mnist.png" width="400" height="350"/>
</center>   

#### Requirement
```
python==3.7.0
numpy==1.18.1
```
#### How to use
Training & Prediction can be run as follows:    
`python train.py train`  
`python train.py predict img.png`  


#### More information
* Please refer to the original paper of ResNet [here](https://arxiv.org/pdf/1512.03385.pdf) for more information.

#### Implementation Notes
* **Note 1**:   
Since datasets are somehow huge and painfully slow in training ,I decided to make number of filters variable. If you want to run it in your PC, you can reduce the number of filters into 32,16,8,6,4 or 2. (64 is by default). For example:  
`model = resnet18.ResNet18((112, 112, 3), classes = 10, filters = 6)`

* **Note 2** :   
You can also make the size of images smaller, so that it can be ran faster and doesn't take too much memories.

### Result for MNIST:   
* epochs = 2
* Filters = 6
* Batch size = 32  
* Optimizer = Adam   
* Learning rate = 0.0001

Name |  Training Accuracy |  Validation Accuracy  |
:---: | :---: | :---:
Resnet18 | 85.52% | 92.29%
Resnet34 | 86.22% | 92.84%
Resnet50 | 92.83% | 94.28%
Resnet101 | 87.93% | 92.64%
Resnet152 | 94.46% | 95.62%
