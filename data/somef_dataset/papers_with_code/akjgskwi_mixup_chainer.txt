# mixup_chainer
implementation of mixup (https://arxiv.org/abs/1710.09412) with Chainer

## mixup
mixup augments training data by making convex combination of them
```math
x = lambda * x_1 + (1-lambda) * x_2
t = lambda * t_1 + (1-lambda) * t_2
```
(`lambda` following Beta(alpha, alpha).)

## Usage
`main.py` execute mixup training using MNIST dataset.
run `python main.py`. you can designate optional arguments. see `python main.py -h`
* `--batchsize`
* `--epoch`
* `--alpha` parameter of Beta distribution(positive value)
* `--decay` parameter of weight decay
* `--lr` learning rate

## Result
```
% python main.py --alpha 0.4
epoch: 5 train_loss: 0.6244 val_loss: 0.1413 val_accuracy: 0.9679
epoch: 10 train_loss: 0.5762 val_loss: 0.1428 val_accuracy: 0.9687
epoch: 15 train_loss: 0.5356 val_loss: 0.1244 val_accuracy: 0.9727
epoch: 20 train_loss: 0.5628 val_loss: 0.1385 val_accuracy: 0.9737
epoch: 25 train_loss: 0.5557 val_loss: 0.1221 val_accuracy: 0.9766
epoch: 30 train_loss: 0.5215 val_loss: 0.1132 val_accuracy: 0.9797
epoch: 35 train_loss: 0.5110 val_loss: 0.0907 val_accuracy: 0.9834
epoch: 40 train_loss: 0.4889 val_loss: 0.0923 val_accuracy: 0.9830
epoch: 45 train_loss: 0.5069 val_loss: 0.0898 val_accuracy: 0.9831
epoch: 50 train_loss: 0.5024 val_loss: 0.0887 val_accuracy: 0.9833
epoch: 55 train_loss: 0.5058 val_loss: 0.0909 val_accuracy: 0.9830
epoch: 60 train_loss: 0.4999 val_loss: 0.0907 val_accuracy: 0.9835
epoch: 65 train_loss: 0.4950 val_loss: 0.0891 val_accuracy: 0.9837
epoch: 70 train_loss: 0.4868 val_loss: 0.0887 val_accuracy: 0.9838
epoch: 75 train_loss: 0.4940 val_loss: 0.0890 val_accuracy: 0.9837
epoch: 80 train_loss: 0.4997 val_loss: 0.0882 val_accuracy: 0.9842
epoch: 85 train_loss: 0.5001 val_loss: 0.0893 val_accuracy: 0.9840
epoch: 90 train_loss: 0.5100 val_loss: 0.0886 val_accuracy: 0.9840
epoch: 95 train_loss: 0.4617 val_loss: 0.0886 val_accuracy: 0.9840
epoch: 100 train_loss: 0.5141 val_loss: 0.0897 val_accuracy: 0.9842
```

loss:
![loss](https://i.imgur.com/CpTmtcM.png)
