# Spectrum Sensing


Forked from [U-Net](https://arxiv.org/abs/1505.04597) in PyTorch, this implantation is used for mixture signal separation and classification

## Usage
**Note : Use Python 3**
### Prediction

To predict and classify a batch of mixture signals

`python predict.py --model MODEL.pth`

You can specify which model file to use with `--model MODEL.pth`.

### Visualization

To visualize the separated component signals, please run visualization.ipynb


### Training

```shell script
> python train.py -h
usage: train.py [-h] [-e E] [-b [B]] [-l [LR]] [-f LOAD]

Train the UNet on 

optional arguments:
  -h, --help            show this help message and exit
  -e E, --epochs E      Number of epochs (default: 5)
  -b [B], --batch-size [B]
                        Batch size (default: 1)
  -l [LR], --learning-rate [LR]
                        Learning rate (default: 0.1)
  -f LOAD, --load LOAD  Load model from a .pth file (default: False)
```

Original paper by Olaf Ronneberger, Philipp Fischer, Thomas Brox: [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)

![network architecture](https://i.imgur.com/jeDVpqF.png)
