# np-transformer

This in an implementation of a single layer of the Transformer module as described in the paper 'Attention is all you need' (Vaswani et al.) [arxiv link][1].

It is a bare-minimum version which I found useful for my purposes.

My main objective was to determine how the self-attention mechanism deals with some toy algorithmic problems and develop intuition on which tasks can or cannot be solved easily.

This version is NOT optimized for performance. Some features are different and some are TODO.

### Summary
- Positional encoding is important (it's quite obvious) to make it work
- It seems to be more robust than other approaches I tried before
- Surprisingly it works with one head and with only one layer, also without having the residual connection and normalization
- Vanilla SGD did not work that well

### Quick start:
```
  python3 transformer.py -t {copy, reverse, rotate, filter} [-l saved_model] [-S sequence length]
```

Example:
```
  python3 transformer.py -t reverse
```

### Some background
Some time ago I was very interested in memory-augmented architectures and their theoretical power when it comes to solving some simple problems requiring manipulating some external storage (Neural Turing Machine). Similarly to a CPU, we could focus on a given memory location based on its address or its content.
Differentiable Neural Computer (DNC) was another incarnation of this idea, well described in the Nature paper[2]. Here's my toy numpy code which tests DNC on a character-level prediction task.

[https://github.com/krocki/dnc](https://github.com/krocki/dnc)

Those architectures where primarily developed with a hope that they could solve problems in an algorithmic way, just like a computer.

I am very interested in this concept and always try to understand if a given problem can be solved in a machine-learning way of seeing input-output examples. I often perform mental experiments in order to determine the number of registers, memory locations, addressing, etc to have a better understanding of what can be learned.

Let's take a program which copies data.

Pseudocode:

```
while (i<N) {
  mem[dst_base+i] = mem[src_base+i];
  i++;
}
```

A program then for something like 'copy' would ignore the content of memory cells, but instead move something from one location to another while advancing the pointer `[i]`. It means that it should be fairly simple to learn such a program, given that the can focus on a location effectively. One major disadvantage of the `1st generation` of the `algorithm-learning` architectures such as NTM or DNC was the fact that the process was sequential in nature and constrained by the recurrent controller. This simple task was actually not that easy since `i` has to be shifted by some precise disrete amount in every iteration. The process also takes N steps.

One concept which actually worked quite well was the content-based addressing mechanism (key-value dot product). It is quite useful if we need to process data based on some content. 

Let's define a task of filtering an array based on a value (let's say we want only even numbers), then the network needs to learn that the values are important as well as their positions.

```
while (i<N) {
  value = mem[src_base+i];
  mem[dst_base+i] = value is odd ? NULL : value;
  i++;
}
```

We can see that we need to 'query' if the value is odd, which in turn can be learned by a network by forming an odd/even mask and the dot product of that mask and a value will indicate if NULL or the value should be written.

### Transformer

The core of the Tranformer module is the dot-product attention which allows us to focus with respect to some value. Here's the crucial part which makes the Transformer different than the previous approaches - We encode the information about position directly into the input stream and therefore we can process the entire input sequence in parallel instead of sequentially.

Let's go back to the copy example. Since all 'i' memory locations are independent, we can copy the data in a single step, assuming no additional constraints. In a way it's a hack, since we mark every item with a tag and later we effectively use content-based addressing to determine the location. It works well in practice.

Here are some experiments I tried and they give some insight into how things are learned.

#### Some experiments

The main thing I was looking at was of course the way self-attention works in those cases.
From the original paper:

<img src=./imgs/attention.png width=500/>

The input to the attention module is `xs` of size N x M, where M is the input(sequence) and N is the attention module dimension. `xs` is fixed, but `vs`, `ks` and `qs` which as arrays for V, K and Q respectively are determined by learnable parameters `Wxv`, `Wxk` and `Wxq`.

```
vs = np.dot(model['Wxv'].T, xs)                                                                            
ks = np.dot(model['Wxk'].T, xs)
qs = np.dot(model['Wxq'].T, xs)
```

I used the original positional encoding.

<img src=./imgs/pe.png width=400/>

#### Copy
Let's take a look at the inputs and learned weights for the copy task.

```
  python3 transformer.py -t copy
```

The images will be generated at the same time when checkpoints are saved (every 100000 iterations or so).
<img src=./imgs/copy.png width=500/>

What we can see in the images above - input looks the same as the output - that's good, we have copied the sequence. In my implementation, I follow the original paper and compute the attention values (size N x N, att_sm in the pdf). Then `attention * vs` gives the output of the module `zs`. The `decoder` in my case takes `zs` and produces `ys = dot(Wzy.T, zs)`, so it's as simple as possible. Then it's normalized by softmax and the loss is based on the cross-entropy.

By observing the input-key weights, we can see that indeed the attention module focuses on the location and ignores the content (upper part of the input vector is the original content, lower part is the positional encoding). `vs` weights on the other hand, learn to retrieve the content and ignore the location information.

#### Filter


```
  python3 transformer.py -t filter
```

This task is the same as copy, but write 'invalid' (value 0) when the value is above some threshold. We can see that both the location and the content are relevant. In the images below, the three initial values are greater than the threshold, we see that it has been correctly filtered and zeros appear on the output. The other 2 values are copied without a change.

<img src=./imgs/filter0.png width=300/>

We can see the clear pattern in weights which mark a mask for the values which are not desired.


<img src=./imgs/filter1.png width=600/>

#### Rotate

```
  python3 transformer.py -t rotate
```

Rotate left and carry the value shifted out into the last position.
```
[a b c d] -> [b c d a]
```

<img src=./imgs/rotate0.png width=300/>

Similar to the copy case, the content is not relevant, which is good from the generalization perpective.

<img src=./imgs/rotate1.png width=600/>

#### Reverse

```
  python3 transformer.py -t reverse
```

Reverse the array.
```
[a b c d] -> [d c b a]
```

<img src=./imgs/reverse0.png width=300/>
<img src=./imgs/reverse1.png width=600/>

[1]: https://arxiv.org/pdf/1706.03762.pdf
