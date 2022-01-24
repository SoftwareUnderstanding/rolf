# Reproducing ***Shift-and-Balance Attention***

First of all, thank the authors very much for sharing this excellent paper ***Shift-and-Balance Attention*** with us. This repository contains SB Attention and simple verification for modified VGG16
with CIFAR10. If there are some bug problems in the 
implementation, please send me an email at yuranusduke@163.com or simply add issue.

## Backgrounds
In this paper, authors rethink the idea from SE net. They argue that SE may make gradient vanishing and lead to the wastage of channels due the use of Sigmoid function, and also it constrains the attention
branch to a certain degree, scaled attention is too sensitive to coordinate and balance the two branches’ contributions. Therefore, they propose Shift-and-Balance(SB) Attention to address above problems. The diagram shows as follows,

![img](./README/sb.png)

And formula goes as follows,

![img](./README/f1.png)

One can read paper carefully to understand how and why they design architecture like this.

## Requirements

```Python
pip install -r requirements.txt 
```

## Implementation

We simply run CIFAR10 with modified VGG16.

### Hyper-parameters and defaults
```bash
--device='cuda' # 'cuda' for gpu, 'cpu' otherwise
--attn_ratio=0.5 # hidden size ratio in the SB or SE operation
--use_sb=True # True to enable SB attention
--use_se=False # True to enable SE attention. Note that use_sb and use_se should not be activated at the same time, they are used to do comparison experiments
--activation='tanh' # activation to do ablation study 'tanh', 'relu', 'sigmoid', 'softmax', 'linear'
--epochs=80 # training epochs
--batch_size=64 # batch size
--init_lr=0.1 # initial learning rate
--gamma=0.2 # learning rate decay
--milestones=[40,60,80] # learning rate decay milestones
--weight_decay=9e-6 # weight decay
```

### Train & Test

```python
    python example.py main \
        --device='cuda' \
        --attn_ratio=0.5 \
        --use_sb=True \
        --use_se=False \
        --activation='tanh' \
        --epochs=80 \
        --batch_size=64 \
        --init_lr=0.1 \
        --gamma=0.2 \
        --milestones=[40,60,80] \
        --weight_decay=9e-6 

```

## Results

### SB & SE
| Model             | Acc.        |
| ----------------- | ----------- |
| baseline       	| 92.27%      |
| SE              	| 92.63%      |
| SB(Sigmoid)       | **92.66%**  |

### Ablation Study for Activations

| Activation        | Acc.        |
| ----------------- | ----------- |
| Tanh            	| 92.22%      |
| Sigmoid         	| **92.66%**  |
| ReLU   			| 92.24%      |
| Softmax   		| 92.30%      |
| Linear   			| 92.18%      |

**Note**: I can't use Tanh in SB to get great performance to surpass SE, but Sigmoid may be promising.

## Training statistics
### Baseline
![img](./README/vgg16_0.5_False_False_tanh.png)

### SE
![img](./README/vgg16_0.5_False_True_tanh.png)

### SB
![img](./README/vgg16_0.5_True_False_tanh.png)

## Paper References
- Shift-and-Balance Attention [[arXiv]](https://arxiv.org/pdf/2103.13080)
- Squeeze-and-Excitation Networks [[arXiv]](https://arxiv.org/abs/1709.01507 )

***<center>Veni，vidi，vici --Caesar</center>***
