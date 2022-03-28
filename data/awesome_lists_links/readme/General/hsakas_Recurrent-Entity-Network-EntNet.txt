# Recurrent Entity Networks

This repository contains an independent TensorFlow implementation of recurrent entity networks from [Tracking the World State with
Recurrent Entity Networks](https://arxiv.org/abs/1612.03969). This paper introduces the first method to solve all of the bAbI tasks using 10k training examples. The author's original Torch implementation is now available [here](https://github.com/facebook/MemNN/tree/master/EntNet-babi).

## Setup

1. Run python main.py to begin training on QA1.
2. To test other records change the dataset_id flag in main.py e.g dataset_id="qa2" to train & test with qa2 set
3. To train with 1k samples, set the flag only_1k=True and to train with 10k samples set flag only_1k=False

## Major Dependencies
- Keras v2.0.4
- Tensorflow v1.2.0rc1

## References
- Mikael Henaff, Jason Weston, Arthur Szlam, Antoine Bordes, and Yann LeCun, "Tracking the World State with Recurrent Entity Networks", arXiv:1612.03969 [cs.CL].
- Jim Fleming's Tensorflow implementation of recurrent entity networks: https://github.com/jimfleming/recurrent-entity-networks


## Results

Percent error for each task within less than 200 epochs, comparing those in the paper to the implementation contained in this repository.

Task | EntNet (paper) | EntNet (repo)
---  | --- | ---
1: 1 supporting fact | 0 | 0
2: 2 supporting facts | 0.1 | 
3: 3 supporting facts | 4.1 | 
4: 2 argument relations | 0 | 0
5: 3 argument relations | 0.3 | 
6: yes/no questions | 0.2 | 50%
7: counting | 0 | 55%
8: lists/sets | 0.5 | 66%
9: simple negation | 0.1 | <1%
10: indefinite knowledge | 0.6 | <1%
11: basic coreference | 0.3 | 0
12: conjunction | 0 | 0
13: compound coreference | 1.3 | 7%
14: time reasoning | 0 | 27%
15: basic deduction | 0 | 
16: basic induction | 0.2 | 
17: positional reasoning | 0.5 | 
18: size reasoning | 0.3 | 
19: path finding | 2.3 | 
20: agents motivation | 0 | 
**Failed Tasks** | 0 | ?
**Mean Error** | 0.5 | ?
