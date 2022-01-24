# RandomMutationSearch
From (ICML) Never Ending Reinforcement-Learning Workshop 2021

This work presents a simple method for self-constructing neural networks via random mutation. It operates in a similar manner to NEAT combined with network pruning; however, changes do not occur on the scale of generations with population search, rather the neural topology is actively modified during the agent's lifetime from experience by deciding whether to accept random mutations. 

The algorithm is as follows:

(1) Initialize and evaluate a network of input and output neurons with no connections and set as current best, \
(2) randomly mutate the current best network structure, \
(3) evaluate network performance over a specified time interval, \
(4) if the observed performance is better than the previous best network (mutation acceptancethreshold) then update best network and performance, \
(5) repeat algorithm starting at step 2.

**Notice that this can be seen as a greedy genetic algorithm with an adaptive child-population size, and a parent-population size of 1.**\
e.g. if N random mutations are denied then the child population size (so far) is N since N mutations of the parent have been evaluated. The greedy aspect comes from the fact that the adaptive population size finishes growing once the first locally optimum solution presents itself.


Cite the paper as:

```
@misc{schmidgall2021selfconstructing,
      title={Self-Constructing Neural Networks Through Random Mutation}, 
      author={Samuel Schmidgall},
      year={2021},
      eprint={2103.15692},
      archivePrefix={arXiv},
      primaryClass={cs.NE}
}
```


Cite code as:

```
@misc{random_mutation_search,
  author = {Samuel Schmidgall},
  title = {Random Mutation Search},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/SamuelSchmidgall/RandomMutationSearch}},
}
```


