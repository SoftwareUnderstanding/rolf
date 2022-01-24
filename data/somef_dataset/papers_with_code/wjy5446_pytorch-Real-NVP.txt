# pytorch-Real-NVP

- simple Real-NVP code
  - prior : Multi-variable Normal Distribution
  - Data : Sklearn moon datasets
- paper : https://arxiv.org/abs/1605.08803

![](./resource/exam1.png)

### Layer

![](./resource/layer.png)



### Loss

![](./resource/loss.png)



### Result

#### Inference p(x) -> P(z)

- **P(x)**

![](./resource/inference1.png)

- **P(z)**

![](./resource/inference2.png)



#### Generate P(z) -> P(x)

- **P(z)**

![](./resource/generate1.png)

- **P(x)**

![](./resource/generate2.png)