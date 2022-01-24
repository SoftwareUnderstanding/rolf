# pytorch-GAN
A PyTorch Implementation of Goodfellow et al.'s Paper on Generative Adversarial Networks. Find the paper at: https://arxiv.org/pdf/1406.2661.pdf

## How to run:
Currently has MNIST experiment implemented. Built with torch 1.1.0 and python3.6.

`pip install -r requirements.txt`

`python train.py --epochs 300 --lr 1e-4 --batch-size 32`

Once train.py is running one can open a new shell and running tensboard in order to track various metrics and current generated images during training.

`tensorboard --logdir=runs/<CURRENT_RUN_DIRECTORY>`

### How to adjust hyperparameters: 
**One can use different arguments defined in train.py to adjust various hyperparameters**

```
--epochs EPOCHS       number of epochs to train for (default: 300)
  --lr LR               learning rate for optimizer (default: 1e-4)
  --batch-size BATCH_SIZE
                        number of examples in a batch (default: 32)
  --device DEVICE       device to train on (default: cuda:0 if cuda is
                        available otherwise cpu)
  --latent-size LATENT_SIZE
                        size of latent space vectors (default: 64)
  --g-hidden-size G_HIDDEN_SIZE
                        number of hidden units per layer in G (default: 256)
  --d-hidden-size D_HIDDEN_SIZE
                        number of hidden units per layer in D (default: 256)
```

## Results:
![Epoch 2](https://i.imgur.com/MbMaKga.png) ![Epoch 20](https://i.imgur.com/W2po4XH.png) ![Epoch 499](https://i.imgur.com/MBs5P0q.png) ![Epoch 999](https://i.imgur.com/gJ2XoPk.png)
