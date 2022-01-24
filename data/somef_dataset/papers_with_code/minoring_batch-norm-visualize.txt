# Experiments of batch normalization using TF 2.0
## Results
Experiment using MNIST dataset using the model paper suggested.
![](https://github.com/minoring/batch-norm-visualize/blob/master/docs/accuracy.png)
## Usage
Using batch normalization.
```
python main.py --epochs=300 --steps_per_epoch=600 --bn=True
```
Without batch normalization.
```
python main.py --epochs=300 --steps_per_epoch=600 --bn=False
```
Tensorboard to visualize
```
tensorboard --logdir=logs
```

## References
### paper
- https://arxiv.org/abs/1502.03167
