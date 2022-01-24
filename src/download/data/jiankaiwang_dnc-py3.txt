# DNC-py3


Differentiable Neural Computer (DNC) is a kind of enhanced neural cell-like LSTM and GRU. It is published in 2016 by Google DeepMind. The main article please refer to https://www.nature.com/articles/nature20101. DNC mainly purposed a new idea to keep memory out of the neural cell but in the external memory. We can train a feedForward or a recurrent neural network to learn how to operate the memory. That is, we are no longer to require RNN cells to keep memorizing and calculating at the same time. 

Such an idea was not the first time to the public, its predecessor NTM was already published in 2014 (https://arxiv.org/pdf/1410.5401.pdf). Compare to NTM, DNC solved three main problems. First NTM cannot guarantee the write head would update the values directly on the correct position, or say it might encounter the interference while writing. Second external memory can not be reused once it had been written. Third NTM writes data in a consecutive way, and it might change to another location while coming to the boundary. This causes a problem while reading consecutive data, the data would be not continuous. DNC provides new mechanisms, including memory matrix, memory usage vector, linking matrix, and precedence vector, etc. to solve the above issues. DNC's architecture is like the below.



![](./docs/imgs/DNC_architecture.png)



In this repository, we are going to demo how to implement such DNC architecture and how to use it in real cases.



## Environment



Scripts are implemented on python3 and tensorflow r1.11 or r1.12. The core scripts are referred to the following repository.

* https://github.com/Mostafa-Samir/DNC-tensorflow
* https://github.com/Kajiyu/dnc-py3
* https://github.com/darksigma/Fundamentals-of-Deep-Learning-Book



### Repoisotry structure

```text
+ dnc-py3
  + dnc
  + docs
  + tasks
    - copy
    - babi
  - README.md
```

The implementation of main DNC architecture is under the folder `dnc-py3` as the package for python. Several tasks are implemented both in training and testing. The scripts are under the folder `tasks`. There are two commonplace tasks, `copy` and `bAbI`.



## Experiments

### Dynamic Memory Mechanisms

This experiment is designed to demonstrate the various functionalities of the external memory access mechanisms such as in-order retrieval and allocation/deallocation.

A similar approach to that of the paper was followed by training a 2-layer feedforward model with only 10 memory locations on a copy task in which a series of 4 random binary sequences each of which is of size 6 (24 piece of information) was presented as input. Details about the training can be found [here](tasks/copy/).

The model was able to learn to copy the input successfully, and it indeed learned to use the mentioned memory mechanisms. The following figure (which resembles **Extended Data Figure 1** in the paper) illustrates that.

*You can re-generate similar figures in the [visualization notebook](tasks/copy/visualization.ipynb)*

![DNC-Memory-Mechanisms](./docs/imgs/DNC-dynamic-mem.png)

- In the **Memory Locations** part of the figure, it's apparent that the model is able to read the memory locations in the same order they were written into.
- In the **Free Gate** and the **Allocation Gate** portions of the figure, it's shown that the free gates are fully activated after a memory location is read and becomes obsolete, while being less activated in the writing phase. The opposite is true for the allocation gate. The **Memory Locations Usage** also demonstrates how memory locations are used, freed, and re-used again time after time.

*The figure differs a little from the one in the paper when it comes to the activation degrees of the gates. This could be due to the small size of the model and the relatively small training time. However, this doesn't affect the operation of the model.*

### Generalization and Memory Scalability

This experiment was designed to check:

- if the trained model has learned an implicit copying algorithm that can be generalized to larger input lengths.
- if the learned model is independent of the training memory size and can be scaled-up with memories of larger sizes.

To approach that, a 2-layer feedforward model with 15 memory locations was trained on a copy problem in which a single sequence of random binary vectors of lengths between 1 and 10 was presented as input. Details of the training process can be found [here](tasks/copy/).

The model was then tested on pairs of increasing sequence lengths and increasing memory sizes with re-training on any of these pairs of parameters, and the fraction of correctly copied sequences out of a batch of 100 was recorded. The model was indeed able to generalize and use the available memory locations effectively without retraining. This is depicted in the following figure which resembles **Extended Data Figure 2** from the paper.

*Similar figures can be re-generated in the [visualization notebook](tasks/copy/visualization.ipynb)*

![DNC-Scalability](./docs/imgs/DNC-scalable.png)

### bAbI Task

This experiment was designed to reproduce the paper's results on the bAbI 20QA task. By training a model with the same parameters as DNC1 described in the paper (Extended Data Table 2) on the **en-10k** dataset, the model resulted in error percentages that *mostly* fell within the 1 standard deviation of the means reported in the paper (Extended Data Table 1). The results, and their comparison to the paper's mean results, are shown in the following table. Details about training and reproduction can be found [here](tasks/babi/).

| Task Name              | Results | Paper's Mean |
| ---------------------- | ------- | ------------ |
| single supporting fact | 0.00%   | 9.0±12.6%    |
| two supporting facts   | 11.88%  | 39.2±20.5%   |
| three supporting facts | 27.80%  | 39.6±16.4%   |
| two arg relations      | 1.40%   | 0.4±0.7%     |
| three arg relations    | 1.70%   | 1.5±1.0%     |
| yes no questions       | 0.50%   | 6.9±7.5%     |
| counting               | 4.90%   | 9.8±7.0%     |
| lists sets             | 2.10%   | 5.5±5.9%     |
| simple negation        | 0.80%   | 7.7±8.3%     |
| indefinite knowledge   | 1.70%   | 9.6±11.4%    |
| basic coreference      | 0.10%   | 3.3±5.7%     |
| conjunction            | 0.00%   | 5.0±6.3%     |
| compound coreference   | 0.40%   | 3.1±3.6%     |
| time reasoning         | 11.80%  | 11.0±7.5%    |
| basic deduction        | 45.44%  | 27.2±20.1%   |
| basic induction        | 56.43%  | 53.6±1.9%    |
| positional reasoning   | 39.02%  | 32.4±8.0%    |
| size reasoning         | 8.68%   | 4.2±1.8%     |
| path finding           | 98.21%  | 64.6±37.4%   |
| agents motivations     | 2.71%   | 0.0±0.1%     |
| **Mean Err.**          | 15.78%  | 16.7±7.6%    |
| **Failed (err. > 5%)** | 8       | 11.2±5.4     |

