# Pytorch Fast GAT Implementation

![Fast GAT](./diagram/fast-gat.png)

This is my implementation of an old paper, [Graph Attention Networks](https://arxiv.org/pdf/1710.10903.pdf).
However, instead of a standard impementation, this one introduces several techniques to speed up the process,
which are found below.

## Installation

```
pip install fast_gat
```

Alternatively,

```
git clone https://github.com/tatp22/pytorch-fast-GAT.git
cd fast_gat
```

## What makes this repo faster?

What is great about this paper is that, besides its state of the art performance on a number of benchmarks,
is that it could be applied to any graph, regardless of its structure. However, this algorithm has a runtime
that depends on the number of edges, and when the graph is dense, this means that it can run in `nodes^2` time.

Most sparsifying techniques for graphs rely on somehow decreasing the number of edges. However, I will try out
a different method: Reducing the number of nodes in the interior representation. This will be done similarly to how
the [Linformer](https://arxiv.org/pdf/2006.04768.pdf) decreases the memory requirement of the internal matrices, which
is by adding a parameterized matrix to the input that transforms it. A challenge here is that since this is a graph,
not all nodes will connect to all other nodes. My plan is to explore techniques to reduce the size of the graph (the
nodes, that is), pass it into the GAT, and then upscale it back to the original size.

Seeing that sparse attention has shown to perfom just as well as traditional attention, could it be the same for graphs?
I will try some experiments and see if this is indeed the case.

This is not yet implemented.

Note: This idea has not been tested. I do not know what its performance will be on real life applications,
and it may or may not provide accurate results.

## Code Example

Right now, there exist two different versions of GAT: one for sparse graphs, and one for dense graphs. The idea in
the end is to use only the dense version, since the sparse version runs slower. It is currently not possible to use
the dense version on very large graphs, since it creates a matrix of size `(n,n)`, which will quickly drain the
system's memory.

As an example, this is how to use the sparse version:

```python
import torch
from fast_gat import GraphAttentionNetwork

nodes = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]], dtype= torch.float)
edges = {0: {1,2}, 1: {0,2,3}, 2: {0,1}, 3: {1}}

depth = 3
heads = 3
input_dim = 3
inner_dim = 2

net = GraphAttentionNetwork(depth, heads, input_dim, inner_dim)

output = net(nodes, edges)
print(output)
```

A point of interest here that one may notice is that the modules assume the graph is directed and that the edges
have already been processed such that the nodes are zero indexed.

## Downsampling method

The first downsampling method that I came up with here takes advantage of the disjoint set data structure, in order
to achieve downsampling in just O(nα(n)) time. This works as follows: until the graph is at the desired number of nodes,
we take an edge u.a.r. from the graph, and use a global `nn.Linear` and run two nodes through it to get an output of one
node, and we replace the starting node with this combination.

In this method, the disjoint set data structure allows preserve our edges such that if nodes `i` and `j` were connected
by a path of length `k` in the original graph `G`, at any point in the downsampling, for our graph `G'`, the nodes `i'`
and `j'` (or whatever they were merged into) are still connected by a path of length `k' <= k`, and the information on
their intermediate connections (if there were any) are stored in the single `nn.Linear` layer, all while keeping efficient
time (less than n^2).

In fact, this method may most likely be parallelizable (best case, O(log(n)α(n)), worst case still O(nα(n))), by choosing
n/2 edges max each step such that each node would have max one edge that is considered and then running O(log(n)) steps
of that downsampling, but for now I will just test the above method.

The downsampling method returns the edges that were merged in order; this makes the upsampling easy, as we just run a
reverse `nn.Linear` that upsamples it from 1 to 2 nodes.

What's nice about this method is that it requires no assumptions on the graph structure.

## Further work that needs to be done

* Test this on a real life graph

## Citation

```
@misc{veličković2018graph,
      title={Graph Attention Networks}, 
      author={Petar Veličković and Guillem Cucurull and Arantxa Casanova and Adriana Romero and Pietro Liò and Yoshua Bengio},
      year={2018},
      eprint={1710.10903},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```
