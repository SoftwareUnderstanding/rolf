# maml
Implementation Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks 
https://arxiv.org/pdf/1703.03400v3.pdf


## Data
Data are available at Kaggle, save three pickled files into the ``data`` folder.
https://www.kaggle.com/whitemoon/miniimagenet

## Requirements
```
python=3.6 (tested)
pytorch=1.6.0
learn2learn=0.1.3
torchvision=0.7.0
```
## Performance

FSL setting for both training and evaluation. 

```
meta-batch = 4 
N way = 5
K shot = 5
Q query = 15
```

Implementation | Dev | Test
--- | --- | ---
Original paper |  | 63.11
Our * | | 


## Acknowledgement

Many thanks to @tristandeleu for his implementations https://github.com/tristandeleu/pytorch-maml

