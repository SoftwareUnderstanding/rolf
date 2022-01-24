# Weight Standardization

[Weight Standardization](https://arxiv.org/abs/1903.10520) (WS) is a normalization method to accelerate **micro-batch training**.
Micro-batch training is hard because small batch sizes are not enough for
training networks with Batch Normalization (BN), while
other normalization methods that do not rely on batch
knowledge still have difficulty matching the performances
of BN in large-batch training.

**Our WS ends this problem** because when used with Group Normalization and trained
with 1 image/GPU, WS is able to match or outperform the
performances of BN trained with large batch sizes with **only
2 more lines of code**.
So if you are facing any micro-batch training problem, please do yourself a favor and try Weight Standardization.
You will be surprised by how well it performs.

<p float="left">
  <img src="imgs/comp.png" height="200" />
  <img src="imgs/norm.png" height="200" />
</p>

## Test Result with Gluon
- TBD

 
## Reference
[Original Repo](https://github.com/joe-siyuan-qiao/WeightStandardization) 
