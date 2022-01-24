# Is feature diversity necessary in the initialization of neural networks ?

This is repository is intended to allow replication of the results presented in the paper:
"Beyond Signal Propagation: Is Feature Diversity Necessary in Deep Neural Network Initialization?"

Code is forked from the repository: https://github.com/valilenk/fixup.git

# Requirements \ Replication details

Main:
* Python (Tested with 3.7.3)
* CUDA (Tested with 10.0.130) + cudnn 7.6.1
* Pytorch (Tested with 1.1.0)

Extra:
* TorchVision  (0.3.0)
* prunhild (0.1.0)
* torchviz (0.0.1)
* numpy (1.16.4)
* Tensorboard (1.13.1, logger: 0.1.0)

run:
`
pip install -r requirements.txt
`

to quickly install the python packages.

# Constnet
The constnet architecture is running by default. 

Usage examples:
```
python train.py --layers 16 --widen-factor 10 --batchnorm   --name constnet -d 0:1 --no-saves --epochs 300 ## Run vanilla constnet on GPU devices 0+1, save no checkpoints.
python train.py --layers 16 --widen-factor 10 --batchnorm   --name constnet_he -d 2:3 --no-saves --varnet --epochs 300 ## Use random inialization (Devices 2+3)
python train.py --layers 16 --widen-factor 10 --batchnorm   --name constnet_deterministic -d 2:3 --no-saves --cudaNoise --epochs 300  ## Deterministic CUDNN
python train.py --layers 16 --widen-factor 10 --batchnorm --droprate 0.01   --name constnet_cuda_dropout -d 0:1 --no-saves --epochs 300   ## Small random dropout
python train.py --layers 16 --widen-factor 10 --batchnorm --droprate 0.01   --name constnet_dropout -d 0:1 --no-saves --epochs 300  --cudaNoise   ## Deterministic + Small random dropout

```

During training, a tensorboard file is generated with the name specified, which includes the forward/backward correlations measured after each epoch.

To view the results, use:

`
tensorboard --logdir runs
`


# LeakyNet

Add the flags: 
`--lrelu 0.01, -a "leakynet"`
to experiment with leakynets.

to control the initialization of each layer, use the parameter:
`--init x_xxxx_xxxx_xxxx` (for a default network of 16 layers)
the name will be matched automatically to match.

where:
* 'h' is for random initialization
* 'i' for identity initialization
* '1' for averaging initialization

Examples:
```
python train.py --layers 16 --widen-factor 10 --batchnorm --lr 0.03 --init 1_ii11_ii11_ii11 -d 0:1 --no-saves -a 'leakynet' --lrelu 0.01
python train.py --layers 16 --widen-factor 10 --batchnorm --lr 0.03 --init h_hhhh_hhhh_hhhh -d 2:3 --no-saves -a 'leakynet' --lrelu 0.01
```

The bash script "lrnet_experiment.sh" can replicate the entire leaky-net experiment:
`sh lrnet_experiment.sh 0:1` 
