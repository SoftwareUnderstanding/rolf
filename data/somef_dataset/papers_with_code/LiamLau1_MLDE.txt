# Solving Differential Equations using Neural Networks


## First Model

* Loss function that takes average for F squared for points sampled from the domain.
    - We follow the methods from [https://arxiv.org/pdf/1711.10561.pdf](https://arxiv.org/pdf/1711.10561.pdf) and [https://arxiv.org/pdf/1902.05563.pdf](https://arxiv.org/pdf/1902.05563.pdf)

* Xavier initialization is such that the initial weights aren't too small or too big. What does this mean? Consider a tanh or sigmoidal activation function. If the weights are too small, near 0 say. The variance of the outputs are small, the activation function will be in the linear regime, not taking the non linear nature. If the weights are too big, the variance over several layer outputs is large, and activation function is usually flat at extremities, therefore not even firing.
    - Methods include: [https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/](https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/) and [https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78](https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78)
    - This [paper](https://arxiv.org/pdf/1502.01852.pdf) gives an example of Xavier and when Xavier doesn't work so well (using ReLu or leaky ReLu activation functions)
    - Xavier uses $Var(W_i) = \frac{2}{n_{in} + n_{out}}$

* Seems to me that the boundary condition part of the loss function is stronger. Less variance therefore that point always is pinned (with enough weights). I've scaled the F squared part of the loss for the differential equation, just multiplied by 10, seems to get better fit. Motivation for this is that part is high variance due to the large number of points, therefore should be weighted higher cost to fit in lost function.
    - I've played around with the scaling of the first term in the loss factor, there seems to be a sweet spot around 100, this should be a parameter?
