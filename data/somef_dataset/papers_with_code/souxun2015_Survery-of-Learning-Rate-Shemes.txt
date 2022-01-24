
# Survery-of-Learning-rate-shemes

## Introduction
This project mainly introduces the learning rate schemes provided by tensorflow and observes their influences on convolutional neural networks. The problem about how they work is not included as it is difficult to explain. Maybe in the future, I will post it once I get them straight. So, there are 15 learning rate schemes we will talk about:
- 1. exponential_decay
- 2. piecewise_constant_decay
- 3. polynominal_decay
- 4. inverse_time_decay
- 5. cosine_decay
- 6. cosine_decay_restarts
- 7. linear_cosine_decay
- 8. noisy_linear_cosine_decay
- 9. tf.train.GradientDescentOptimizer
- 10. tf.train.MomentumOptimizer
- 11. tf.train.AdamOptimizer // tf.train.AdagradOptimizer // tf.train.AdadeletaOptimizer // tf.train.AdagradDAOptimizer
- 12. tf.train.RMSPropOptimizer
- 13. tf.train.FtrlOptimizer
We conduct experiments on Cifar10 with these shemes, and then make analyses on different combinations among them.

### Learning Rate Decay Schemes
- 1. exponential_decay
```
# python
decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
If the argument `staircase` is `True`, then `global_step / decay_steps` is an integer division and the decayed learning rate follows a staircase function.
```
- 2. piecewise_constant_decay
```
Example: use a learning rate that's 1.0 for the first 100001 steps, 0.5 for the next 10000 steps, and 0.1 for any additional steps.
```
- 3. polynomial_decay
```
global_step = min(global_step, decay_steps)
decayed_learning_rate = (learning_rate - end_learning_rate) *
(1 - global_step / decay_steps) ^ (power) +
end_learning_rate

If `cycle` is True then a multiple of `decay_steps` is used, the first one that is bigger than `global_steps`.

decay_steps = decay_steps * ceil(global_step / decay_steps)
decayed_learning_rate = (learning_rate - end_learning_rate) *
(1 - global_step / decay_steps) ^ (power) + end_learning_rate
```
- 4. inverse_time_decay
```
When training a model, it is often recommended to lower the learning rate as the training progresses.  This function applies an inverse decay function to a provided initial learning rate.

decayed_learning_rate = learning_rate / (1 + decay_rate * global_step / decay_step)
```
- 5. cosine_decay
```
See [Loshchilov & Hutter, ICLR2016], SGDR: Stochastic Gradient Descent with Warm Restarts. https://arxiv.org/abs/1608.03983

When training a model, it is often recommended to lower the learning rate as the training progresses. This function applies a cosine decay function to a provided initial learning rate.

global_step = min(global_step, decay_steps)
cosine_decay = 0.5 * (1 + cos(pi * global_step / decay_steps))
decayed = (1 - alpha) * cosine_decay + alpha
decayed_learning_rate = learning_rate * decayed
```
- 6. cosine_decay_restarts
```
See [Loshchilov & Hutter, ICLR2016], SGDR: Stochastic Gradient Descent with Warm Restarts. https://arxiv.org/abs/1608.03983

When training a model, it is often recommended to lower the learning rate as the training progresses. This function applies a cosine decay function with restarts to a provided initial learning rate.

The function returns the decayed learning rate while taking into account possible warm restarts. The learning rate multiplier first decays from 1 to `alpha` for `first_decay_steps` steps. Then, a warm
restart is performed. Each new warm restart runs for `t_mul` times more steps and with `m_mul` times smaller initial learning rate.
```
- 7. linear_cosine_decay
```
See [Bello et al., ICML2017] Neural Optimizer Search with RL.
https://arxiv.org/abs/1709.07417

For the idea of warm starts here controlled by `num_periods`,
see [Loshchilov & Hutter, ICLR2016] SGDR: Stochastic Gradient Descent with Warm Restarts. https://arxiv.org/abs/1608.03983
Note that linear cosine decay is more aggressive than cosine decay and larger initial learning rates can typically be used.
```
- 8. noisy_linear_cosine_decay
```
When training a model, it is often recommended to lower the learning rate as the training progresses.  This function applies a noisy linear cosine decay function to a provided initial learning rate.

global_step = min(global_step, decay_steps)
linear_decay = (decay_steps - global_step) / decay_steps)
cosine_decay = 0.5 * (
1 + cos(pi * 2 * num_periods * global_step / decay_steps))
decayed = (alpha + linear_decay + eps_t) * cosine_decay + beta
decayed_learning_rate = learning_rate * decayed

where eps_t is 0-centered gaussian noise with variance
initial_variance / (1 + global_step) ** variance_decay
```
### Optimizer Schemes
This is  a great post that gives a comprehensive introduction to the optimizer schemes.
http://ruder.io/optimizing-gradient-descent/index.html#fn25 
- 9. tf.train.GradientDescentOptimizer
This is original optimizer, the gradient is just based on the current batch.
- 10. tf.train.MomentumOptimizer
This optimizer contains a momentum to update the gradients. It means that updating the gradient is relationed to the previous batches.
In the its inputs, there is a switch to control how to update the variables(original Momentum or Nesterov Momentum)
- 11. tf.train.AdamOptimizer // tf.train.AdagradOptimizer // tf.train.AdadeletaOptimizer // tf.train.AdagradDAOptimizer

- 12. tf.train.RMSPropOptimizer
```
__init__(
learning_rate,
decay=0.9,
momentum=0.0,
epsilon=1e-10,
use_locking=False,
centered=False,
name='RMSProp'
)
```
- 13. tf.train.FtrlOptimizer
```
__init__(
learning_rate,
learning_rate_power=-0.5,
initial_accumulator_value=0.1,
l1_regularization_strength=0.0,
l2_regularization_strength=0.0,
use_locking=False,
name='Ftrl',
accum_name=None,
linear_name=None,
l2_shrinkage_regularization_strength=0.0
)
```
## Comparable Analyses
- 1. 
