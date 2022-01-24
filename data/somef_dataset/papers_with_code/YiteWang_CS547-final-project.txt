# CS547 final project

* Team members: Yite Wang (yitew2) , Jing Wu(jingwu6) , Yuchen He(he44), Randy Chase (randyjc2)

### Code structure

The code consists of  5 python files, which are:

* `Arch.py`: It contains all the functions/classes to create the discriminator and generators. Here we only support resnet generator and 70X70 patch discriminator.

* `Main.py`: It is the main file that takes in all the arguments including all the hyperparameters.

* `Utils.py`: It contains several functions used in training. They include set neural network not update, sample buffer, learning rate scheduler and initialization function for neural networks.

* `CycleGAN.py`: This is the most important part of the code that contains a class which defines the whole training process of cycleGAN. In initialization part, all the neural networks, optimizers and schedulers are created. In start_train() function,   it first loads all the data and first update generator. In the generator training phase, we turned off gradient calculations of discriminators to make computation faster. After that we turned on gradient calculation of discriminators and then update discriminators. The last part of it is saving all the models and losses after every certain number of epochs.


### How to use the Code

Clone this repo to your machine.

Create folder `datasets` and put the dataset you want into the folder `datasets`.

Notice the structure under `datasets` is as follows:

```
vangogh2photo
│
└───TrainA
│   │   
│   └───Apple_train
│       │   pic1.png
│       │   pic2.png
│       │   ...
│    
│   
└───TrainB
│   │   
│   └───Orange_train
│       │   pic1.png
│       │   pic2.png
│       │   ...
│    
│   
└───TestA
│   │   
│   └───Apple_test
│       │   pic1.png
│       │   pic2.png
│       │   ...
│    
│   
└───TestB
    │   
    └───Orange_test
        │   pic1.png
        │   pic2.png
        │   ...
```

An example modified dataset can be downloaded [here](https://drive.google.com/open?id=1-t9Q2kMwcPxdUe-v6Gy_Kg3LaP68F27K)

Then run the following code in terminal to train:

`python main.py --epochs 200 --decay_epoch 100 --batch_size 2 --training True --testing True --data_name apple2orange`

If you only want to test:

`python main.py --test_batch_size 1 --testing True --data_name apple2orange`

If you want to do Monet, you should add identity loss, which needs extra arguments: `--use_id_loss True`

For more information, check `main.py` or run the following code:

`python main.py -h`

### Reference:

1.Original CycleGAN paper: [arXiv](https://arxiv.org/abs/1703.10593)

2.Original CycleGAN repo: [repo](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

3.Simple implementation of CycleGAN: [repo](https://github.com/arnab39/cycleGAN-PyTorch)
