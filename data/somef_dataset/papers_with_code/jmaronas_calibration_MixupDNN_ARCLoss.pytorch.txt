# On Calibration of Mixup Training for Deep Neural Networks
This repository contains the code and additional details for the work analyzing and improving calibration properties of Mixup which is available here [https://arxiv.org/abs/2003.09946](https://arxiv.org/abs/2003.09946) . Details on the individual values of all the models trained in this work, plus specific hyperparameters of the proposed loss function can be found in [this file](./SSPR_appendix_Github.pdf) attached in this repository.. Details on specific hyperparameters such as weight decay and so on are provided in the code.

The topologies of the networks implemented and their hyperparameters are taken from these three Githubs:

* https://github.com/kuangliu/pytorch-cifar

* https://github.com/meliketoy/wide-resnet.pytorch

* Pre-Trained Models provided by PyTorch API https://pytorch.org/docs/stable/torchvision/models.html

## Install

Assuming that anaconda is installed in `/usr/local/anaconda3/` execute `./install.sh` 

## Reproducing Results from the Paper

To reproduce some of the results  in the paper execute the following command in the code folder:

```./launch_examples.sh``` 

For Birds and Cars dataset using big networks (such as the DenseNet-121) you need at least 20-24 GB of GPU memory; as this datasets have images with ImageNet size (224x224xRGB).

Moreover note that exact reproducibility cannot be ensure as CuDNN library does not guarantee  exact reproducibility and it is necessary for run all the models of this work in a reasonable time, see [https://pytorch.org/docs/stable/notes/randomness.html](https://pytorch.org/docs/stable/notes/randomness.html)

 ## Launch Models

To launch the models in the paper or launch models in general we provide several files:

#### Baseline Models

```python main_baseline.py  --model_net resnet-18 --dataset cifar10 --n_gpu 0 1 --dropout 0.0 --use_valid_set 0```

* ```--n_gpu``` is used to select the GPU devices where it is launch. In this case 0 1 willl launch the model distributed according to gpus labelled with 0 and 1 in your machine. 
* ```--use_valid_set``` use a validation set (1) or not (0).

#### Baseline Models with Mixup

```python main_baseline_mixup.py  --model_net resnet-18 --dataset birds --mixup_coeff 0.4 --n_gpu 0 1 --dropout 0.0 --use_valid_set 0```

* where mixup_coeff specifies the value of $\lambda$ in the original work, available here [https://arxiv.org/abs/1710.09412](https://arxiv.org/abs/1710.09412)

#### MMCE calibration technique

To launch the calibration technique we compare with, available here [http://proceedings.mlr.press/v80/kumar18a/kumar18a.pdf](http://proceedings.mlr.press/v80/kumar18a/kumar18a.pdf), launch the following file:

```python main_baseline_MMCE.py  --model_net resnet-18 --dataset cifar10 --n_gpu 0 1 --dropout 0.0 --use_valid_set 0 --lamda 0.5```

* where ```--lambda``` specifies the value of $\lambda$ in the equation number (4) of the above reference.

#### MMCE calibration over Mixup

To launch this calibration technique over a Mixup trained DNN launch the following file:

```python main_mixup_MMCE.py     --model_net resnet-18 --dataset cifar10 --mixup_coeff 1.0 --n_gpu 0 1 --dropout 0.0 --use_valid_set 0 --lamda 0.1 --cost_over_mix_image 0```

* where cost_over_mix_image states if the loss is applied to the mixup image $\mathtt{MMCE}(\tilde{x})$ (1) or it is applied separately over different images $\mathtt{MMCE}(x_1,x_2)$   

#### ARC Loss

To launch the ARC loss execute:

```python main_baseline_ARC.py  --model_net resnet-18 --dataset cifar10 --n_gpu 0 1 --dropout 0.0 --use_valid_set 0 --cost_type 'square[avgconf_sub_acc]' --bins_for_train 1 --lamda 1.0```

* where ```--lamda``` is the $\beta$ of our paper.
* ```--cost_type``` is the two variants of our loss being  $V_1$='square[avgconf_sub_acc]'  and $V_2$='avg[square[conf_sub_acc]]'
* ```--bins_for_train``` is the value of $M$ in the paper i.e the number of bins in which the confidence range is divided.

#### ARC Loss over Mixup

To launch the ARC loss over a Mixup trained DNN launch the following file:

```python main_mixup_ARC.py     --model_net resnet-18 --dataset cifar10 --mixup_coeff 1.0 --n_gpu 0 1 --dropout 0.0 --use_valid_set 0 --cost_type 'square[avgconf_sub_acc]' --lamda 4 --bins_for_train 1 --cost_over_mix_image 0```

#### Cross Entropy over Train and ARC over Validation

To launch the experiments where the ARC loss is applied over the validation set and the CE over the train set (table 5 in appendix D)  launch the following file:

```python main_valid.ARC_train.CE.py --model_net resnet-18 --dataset cifar10 --n_gpu 0 1 --dropout 0.0 --use_valid_set 1 --cost_type 'square[avgconf_sub_acc]' --lamda 0.0 --bins_for_train 1```

