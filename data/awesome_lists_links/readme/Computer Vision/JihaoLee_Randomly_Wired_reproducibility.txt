# Randomly_Wired_reproducibility
This is a reimplementation of [Exploring Randomly Wired Neural Networks for Image Recognition](https://arxiv.org/abs/1904.01569)

# Requirements
* Python 3.5
* PyTorch==1.0.0
* sklearn, tensorboardX, numpy

# Usage
generate random graph
----------
ws: `python graph/ws.py -k -p`  
er: `python graph/er.py -p`  
ba: `python graph/ba.py -m`  

train
----------
```
python train.py --data <path to ImageNet>  
                --regime <small is True, regular is False>  
                --base_channels <78, 109, 154>
```

eval
----------
```
python eval.py --data <path to ImageNet>  
               --regime <small is True, regular is False>  
               --base_channels <78, 109, 154>  
               --model_path <path to trained path
```
                
# Results
Validation result on Imagenet(ILSVRC2012) dataset:

| Top 1 accuracy (%)         | Paper | Here |
| -------------------------- | ----- | ---- |
| RandWire-WS(4, 0.75), C=78 | 74.7  | 70.0 |

# Citation
Xie S, Kirillov A, Girshick R, et al. Exploring randomly wired neural networks for image recognition[J]. arXiv preprint arXiv:1904.01569, 2019.  
[Seungwon Park](https://github.com/seungwonpark/RandWireNN)
