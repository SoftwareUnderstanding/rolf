# real_nvp

This is pytorch implementation of Real NVP[1] paper for toy dataset.

## Environment 
* python 3
* pytorch 3.6

## Implementation Results
| Ground Truth x | predicted x |
|:-:|:-:|
|![GT x](/img/train_x.jpg)|![x'](/img/pred_x.jpg)|

| z' = f(x) | z ~ p(z) |
|:-:|:-:|
|![z'](/img/train_z.jpg)|![z](/img/z.jpg)|

![loss_graph](/img/loss_graph.png)

[1] Density estimation using Real NVP(https://arxiv.org/abs/1605.08803)
