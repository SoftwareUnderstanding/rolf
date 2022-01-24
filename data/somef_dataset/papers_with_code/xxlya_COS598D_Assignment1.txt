# Network Pruning
### Assignment 1 for COS598D: System and Machine Learning

In this assignment, you are required to evaluate three advanced neural network pruning methods, including SNIP [1], GraSP [2] and SynFlow [3], and compare with two baseline pruning methods,including random pruning and magnitude-based pruning. In `example/singleshot.py`, we provide an example to do singleshot global pruning without iterative training. In `example/multishot.py`, we provde an example to do multi-shot iterative training. This assignment focuses on the pruning protocol in `example/singleshot.py`. Your are going to explore various pruning methods on differnt hyperparameters and network architectures.

***References***

[1] Lee, N., Ajanthan, T. and Torr, P.H., 2018. Snip: Single-shot network pruning based on connection sensitivity. arXiv preprint arXiv:1810.02340.

[2] Wang, C., Zhang, G. and Grosse, R., 2020. Picking winning tickets before training by preserving gradient flow. arXiv preprint arXiv:2002.07376.

[3] Tanaka, H., Kunin, D., Yamins, D.L. and Ganguli, S., 2020. Pruning neural networks without any data by iteratively conserving synaptic flow. arXiv preprint arXiv:2006.05467.

### Additional reading materials:

A recent preprint [4] assessed [1-3].

[4] Frankle, J., Dziugaite, G.K., Roy, D.M. and Carbin, M., 2020. Pruning Neural Networks at Initialization: Why are We Missing the Mark?. arXiv preprint arXiv:2009.08576.

## Getting Started
First clone this repo, then install all dependencies
```
pip install -r requirements.txt
```

## How to Run 
Run `python main.py --help` for a complete description of flags and hyperparameters. You can also go to `main.py` to check all the parameters. 

Example: Initialize a VGG16, prune with SynFlow and train it to the sparsity of 10^-0.5 . We have sparsity = 10**(-float(args.compression)).
```
python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner synflow --compression 0.5
```

To save the experiment, please add `--expid {NAME}`. `--compression-list` and `--pruner-list` are not available for runing singleshot experiment. You can modify the souce code following `example/multishot.py` to run a list of parameters. `--prune-epochs` is also not available as it does not affect your pruning in singleshot setting. 

For magnitude-based pruning, please set `--pre-epochs 200`. You can reduce the epochs for pretrain to save some time. The other methods do pruning before training, thus they can use the default setting `--pre-epochs 0`.

Please use the default batch size, learning rate, optimizer in the following experiment. Please use the default training and testing spliting. Please monitor training loss and testing loss, and set suitable training epochs. You may try `--post-epoch 100` for Cifar10 and `--post-epoch 10` for MNIST.

## You Tasks

### 1. Hyper-parameter tuning

#### Testing on different archietectures. Please fill the results table:
*Test accuracy (top 1)* of pruned models on CIFAR10 and MNIST (sparsity = 10%). `--compression 1` means sparsity = 10^-1.
```
python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner synflow --compression 1
```
```
python main.py --model-class default --model fc --dataset cifar10 --experiment singleshot --pruner synflow --compression 1
```
|   Data  |   Arch |   Rand |  Mag |  SNIP |  GraSP | SynFlow       |   
|----------------|----------------|-------------|-------------|-------------|---------------|----------------|
|Cifar10 | VGG16 |    |      |        |     |         |
|MNIST| FC |    |      |        |      |         |


#### Tuning compression ratio. Please fill the results table:
Prune models on CIFAR10 with VGG16, please replace {} with sparsity 10^-a for a \in {0.05,0.1,0.2,0.5,1,2}. Feel free to try other sparsity values.

```
python main.py --model-class lottery --model vgg16 --dataset cifar10 --experiment singleshot --pruner synflow  --compression {}
```
***Testing accuracy (top 1)***

|   Compression |   Rand |  Mag |  SNIP |  GraSP | SynFlow       |   
|----------------|-------------|-------------|-------------|---------------|----------------|
| 0.05|    |      |        |     |         |
| 0.1|    |      |        |     |         |
| 0.2|    |      |        |      |         |
| 0.5|    |      |        |      |         |
| 1|    |      |        |      |         |
| 2|    |      |        |      |         |

***Testing time (infernce on testing dataset)***

|   Compression |   Rand |  Mag |  SNIP |  GraSP | SynFlow       |   
|----------------|-------------|-------------|-------------|---------------|----------------|
| 0.05|    |      |        |     |         |
| 0.1|    |      |        |     |         |
| 0.2|    |      |        |      |         |
| 0.5|    |      |        |      |         |
| 1|    |      |        |      |         |
| 2|    |      |        |      |         |

To track the runing time, you can use `timeit`. `pip intall timeit` if it has not been installed.
```
import timeit

start = timeit.default_timer()

#The module that you try to calculate the running time

stop = timeit.default_timer()

print('Time: ', stop - start)
```


***FLOP***

|   Compression |   Rand |  Mag |  SNIP |  GraSP | SynFlow       |   
|----------------|-------------|-------------|-------------|---------------|----------------|
| 0.05|    |      |        |     |         |
| 0.1|    |      |        |     |         |
| 0.2|    |      |        |      |         |
| 0.5|    |      |        |      |         |
| 1|    |      |        |      |         |
| 2|    |      |        |      |         |

For better visualization, you are encouraged to transfer the above three tables into curves and present them as three figrues.
### 2. The compression ratio of each layer
Report the sparsity and draw the weight histograms of each layer using pruner Rand |  Mag |  SNIP |  GraSP | SynFlow with the following settings
`model = vgg16`, `dataset=cifar10`, `compression = 0.5`

***Bonus (optional)***

Report the FLOP of each layer using pruner Rand |  Mag |  SNIP |  GraSP | SynFlow with the following settings
`model = vgg16`, `dataset=cifar10`, `compression= 0.5`.
### 3. Explain your results and submit a short report.
Please describe the settings of your experiments. Please include the required results (described in Task 1 and 2). Please add captions to describe your figures and tables. It would be best to write brief discussions on your results, such as the patterns (what and why), conclusions, and any observations you want to discuss.  
