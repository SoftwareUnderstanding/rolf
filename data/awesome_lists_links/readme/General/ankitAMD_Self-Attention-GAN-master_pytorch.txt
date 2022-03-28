# Self-Attention GAN for NON-CUDA / NON-GPU USER in Pytorch (i talking about code )#####

MUST CHECKOUT LAST BELOWS POINT IT MAY BE VERY USEFUL FOR YOU.............




**[Han Zhang, Ian Goodfellow, Dimitris Metaxas and Augustus Odena, "Self-Attention Generative Adversarial Networks." arXiv preprint arXiv:1805.08318 (2018)](https://arxiv.org/abs/1805.08318).**

## Meta overview
This repository provides a PyTorch implementation of [SAGAN](https://arxiv.org/abs/1805.08318). Both wgan-gp and wgan-hinge loss are ready, but note that wgan-gp is somehow not compatible with the spectral normalization. Remove all the spectral normalization at the model for the adoption of wgan-gp.

Self-attentions are applied to later two layers of both discriminator and generator.

<p align="center"><img width="100%" src="image/main_model.PNG" /></p>

## Current update status
* [ ] Supervised setting
* [ ] Tensorboard loggings
* [x] **[20180608] updated the self-attention module. Thanks to my colleague [Cheonbok Park](https://github.com/cheonbok94)! see 'sagan_models.py' for the update. Should be efficient, and run on large sized images**
* [x] Attention visualization (LSUN Church-outdoor)
* [x] Unsupervised setting (use no label yet) 
* [x] Applied: [Spectral Normalization](https://arxiv.org/abs/1802.05957), code from [here](https://github.com/christiancosgrove/pytorch-spectral-normalization-gan)
* [x] Implemented: self-attention module, two-timescale update rule (TTUR), wgan-hinge loss, wgan-gp loss

&nbsp;
&nbsp;

## Results

### Attention result on LSUN (epoch #8)
<p align="center"><img width="100%" src="image/sagan_attn.png" /></p>
Per-pixel attention result of SAGAN on LSUN church-outdoor dataset. It shows that unsupervised training of self-attention module still works, although it is not interpretable with the attention map itself. Better results with regard to the generated images will be added. These are the visualization of self-attention in generator layer3 and layer4, which are in the size of 16 x 16 and 32 x 32 respectively, each for 64 images. To visualize the per-pixel attentions, only a number of pixels are chosen, as shown on the leftmost and the rightmost numbers indicate. 

### CelebA dataset (epoch on the left, still under training)
<p align="center"><img width="80%" src="image/sagan_celeb.png" /></p>

### LSUN church-outdoor dataset (epoch on the left, still under training)
<p align="center"><img width="70%" src="image/sagan_lsun.png" /></p>

## Prerequisites
* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.3.0](http://pytorch.org/)

&nbsp;

## Usage

#### 1. Clone the repository
```bash
$ git clone https://github.com/heykeetae/Self-Attention-GAN.git
$ cd Self-Attention-GAN
```

#### 2. Install datasets (CelebA or LSUN)
```bash
$ bash download.sh CelebA
or
$ bash download.sh LSUN
```


#### 3. Train 
##### (i) Train
```bash
$ python python main.py --batch_size 64 --imsize 64 --dataset celeb --adv_loss hinge --version sagan_celeb
or
$ python python main.py --batch_size 64 --imsize 64 --dataset lsun --adv_loss hinge --version sagan_lsun
```
#### 4. Enjoy the results
```bash
$ cd samples/sagan_celeb
or
$ cd samples/sagan_lsun

```

SOME IMPORTANT POINTS

1.  Samples generated every 10(in parent file its 100) iterations are located. The rate            of sampling could be controlled via --sample_step (ex, --sample_step 10).

2.  #code->> parser.add_argument('--total_step', type=int, default=100, help='how many times to update the generator')  which are used in 

parameter.py file and  update the generator after 100 --totalstep but in its parent file 

it originally is default =1000000, so i change it default=100 that  its possible for the 

person which have not GPU .

3. In This code i removed cuda because a new coder or non-coder or low middle class 

people have not basically have'nt afford GPU so they 

also experience without gpu and seen what are the result or changes or inference are 

come after train and learn a lot.

4. I do some chnges like removal of Cuda because we require GPU (which are costly but 

more effective and give more speed of our system and train a lot more faster than CPU 

and taking very less time.)

5. In above ### parameter.py #### file we use or import argparse (# import argpase) and 

use argument to intialize our variable/ argument module (like variable as understandable

term it may be hyperparameter or anything also) by default value and what it type and 

also give them choice and also automatically generates help and usage messages and issue 

errors when user give the program invalid argument.(pythonforbegginer.com/argparse/argparse.tutorial ).

6. ## super function() ##used in program go this link for explain... 

--(https://www.pythonforbeginners.com/super/working-python-super-function)

7. transform function in data_loader.py used for augmentation of dataset.

8. PARENT file ----(https://github.com/heykeetae/Self-Attention-GAN)

 where i learn but i do changes and make better for non cuda user please give star on my 
 
 link if you like .
 
 8.Pytorch is a Deep learning framework which are comes mixing of python and "torch"

torch framework comes from  #lua progamming language# basically which are mainly 

originated for research purpose for new model comes and easily deploy and use and find 

the result/INference.
 
 #INFERENCE---------------------------------------------------------------------------------

Inference means estimating the values of some (usually hidden random) variable given some observation.

 i think there isn’t much of a difference (at least conceptually) between infernce and training.
 
 Deep learning is revolutionizing many areas of machine perception, with the potential to impact the everyday experience of 
 
 people everywhere. On a high level, working with deep neural networks is a two-stage process: First, a neural network is 
 
 trained: its parameters are determined using labeled examples of inputs and desired output. Then, the network is deployed to run 
 
 inference, using its previously trained parameters to classify, recognize and process unknown inputs.
 
 https://devblogs.nvidia.com/wp-content/uploads/2015/08/training_inference1.png
 
 Deep Neural Network Training vs. Inference

Figure 1: Deep learning training compared to inference. In training, many inputs, often in large batches, are used to train a 

deep neural network. In inference, the trained network is used to discover information within new inputs that are fed through the 

network in smaller batches.

https://devblogs.nvidia.com/inference-next-step-gpu-accelerated-deep-learning/

https://www.quora.com/What-is-the-difference-between-inference-and-prediction-in-machine-learning
