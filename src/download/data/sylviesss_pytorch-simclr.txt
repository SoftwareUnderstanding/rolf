# (Probabilistic) SimCLR

In this project, we present an implementation of [SimCLR](https://arxiv.org/abs/2002.05709) in PyTorch. We also form
probabilistic notion of the contrastive learning framework and derive a new loss function. The goal is to truly 
understand how a contrastive learning model (SimCLR) learns, how to interpret learned representations, and to quantify
and interpret uncertainty.     

## Run 
To pretrain the model with gradient accumulation with batch size = `n_accum * 64`, for a number of epochs = 
`num_of_epochs`, dataset = `"cifar10"/"stl10"`, path for saving the model and checkpoints = `"/path/for/saving/"`, use_new_loss, run
```
python3 pretrain.py --n_epoch=num_of_epochs --accum_steps=n_accum --dataset=dataset --path_for_saving="/path/for/saving/" --new_loss=use_new_loss
```

## Data 
The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset is used to build a prototype. 

Below is a sample of augmented CIFAR-10 data:
![Augmented Pairs of Images for Constrastive Learning](examples/augmented_samples.PNG)
 

## Methodologies 
#### Augmentation
Augmentation performed for this project followed exactly the same procedure as what was carried out in the paper. 

For clarity, we list the steps here:
- Random cropping (inception-style: random crop size uniform from 0.08 to 1 in area and a random aspect ratio) and 
resizing to original size with random flipping (p=50%); `torchvision.transforms.RandomResizedCrop`
- Random color distortions (color jittering + color dropping) `transforms.ColorJitter`, `transforms.RandomApply`, 
`transforms.RandomGrayscale`
- Random Gaussian blur (p=50%). Randomly sample volatility in [0.1, 2.0], and the kernel size is 10% of the image 
height/width.


#### Encoder
Following the paper, we used a slightly modified `ResNet50` as the encoder for CIFAR images. 

We modify the original resnet module in [pytorch](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py) 
by defining a wrapper on top of it in order to: 
- replace the first 7x7 convolution layer of stride 2 (`conv1`) with a 3x3 convolution of stride 1, to adjust for 
the smaller resolution of images in CIFAR10.
- remove the first maxpooling operation (`maxpool`)
- remove the last fully-connected layer and take the output of the average pooling layer


#### Main Model
The basic model for pretraining consists of:
- Encoder `f` (`ResnetEncoder`)
- Projection head `g` (2-layer (could adjust this in further experiments) MLP with a RELU activation and batch 
normalization)


#### (Pre)Training
-  Currently using a `batch_size` of 512 and gradient accumulation to allow (relatively) larger batch training 
on a single Nvidia Tesla GPU with 12GB RAM. 
- **Different from the paper**, [Adam](https://pytorch.org/docs/stable/_modules/torch/optim/adam.html#Adam) 
was used with a learning rate of 1e-3 and a `weight_decay` of 1e-6 in pretraining. [Lars](https://arxiv.org/pdf/1708.03888.pdf)
optimizer could be added later. 


#### Linear Evaluation
After pretraining the model, features are extracted with the model and used as input to a linear classifier trained 
with `CrossEntropyLoss`. This linear model is trained with the `L-BFGS` optimizer as suggested in the paper. 
 

#### Semi-Supervised Learning
During fine-tuning, we copy the pretrained model weights to a new model, remove the projection heads and attach a linear
classifier. This new model is trained with a fraction (e.g., 10%) of labelled data using `SGD` with Nesterov momentum. 


## Cite 
```
@article{chen2020simple,
  title={A Simple Framework for Contrastive Learning of Visual Representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  journal={arXiv preprint arXiv:2002.05709},
  year={2020}
}
