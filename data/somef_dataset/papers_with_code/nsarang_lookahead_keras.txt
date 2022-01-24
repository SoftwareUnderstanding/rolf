# Lookahead Optimizer

## Reference
Michael R. Zhang, James Lucas, Geoffrey Hinton, Jimmy Ba. **Lookahead Optimizer: k steps forward, 1 step back.** [[Arxiv](https://arxiv.org/abs/1907.08610)]

## Usage
```python
from keras.optimizers import SGD
from lookahead import Lookahead

sgd = SGD(lr=0.001)
lookopt = Lookahead(optimizer=sgd, k=3, alpha=0.5)

model.compile(optimizer=lookopt, ...)
```
