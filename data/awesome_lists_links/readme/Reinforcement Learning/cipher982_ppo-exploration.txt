[//]: # (Image References)

[image1]: https://raw.githubusercontent.com/cipher982/ppo-exploration/master/images/pong_screenshot_small.png "Pong Screenshot"

[pimg1]: https://raw.githubusercontent.com/cipher982/ppo-exploration/master/images/pimg1.png
[pimg2]: https://raw.githubusercontent.com/cipher982/ppo-exploration/master/images/pimg2.png
[pimg3]: https://raw.githubusercontent.com/cipher982/ppo-exploration/master/images/pimg3.png
[pimg4]: https://raw.githubusercontent.com/cipher982/ppo-exploration/master/images/pimg4.png
[pimg5]: https://raw.githubusercontent.com/cipher982/ppo-exploration/master/images/pimg5.png
[pimg6]: https://raw.githubusercontent.com/cipher982/ppo-exploration/master/images/pimg6.png
[pimg7]: https://raw.githubusercontent.com/cipher982/ppo-exploration/master/images/pimg7.png
[pimg8]: https://raw.githubusercontent.com/cipher982/ppo-exploration/master/images/pimg8.png
[pimg9]: https://raw.githubusercontent.com/cipher982/ppo-exploration/master/images/pimg9.png
[pimg10]: https://raw.githubusercontent.com/cipher982/ppo-exploration/master/images/pimg10.png
[pimg11]: https://raw.githubusercontent.com/cipher982/ppo-exploration/master/images/pimg11.png



# Proximal Policy Optimization Notes
#### David Rose
#### 2018-12-28
### Introduction

![image1]

## **Proximal Policy Optimization**


**SUMMARY**

1. Collect trajectories based on PIE THETA, initialize theta&#39;=theta
2. Compute gradient of clipped surrogate function using the trajectories
3. Update theta' using gradient ascent
4. Repeat steps 2-3 without generating new trajectories (a few times maybe)
5. Set new policies (theta=theta&#39;) and go back to step 1, repeat.



## **ARXIV paper (openAI)**
https://arxiv.org/abs/1707.06347
- Alternate through sampling data through interaction, and optimizing a _surrogate_ objective function using SGD.
- **OLD** :
  - Typically, policy methods perform one gradient update per data sample
  - TYPES:
    - DQN, policy gradients, trust regions
    - DQN: bad on continuous spaces
  - Room for improvement with less hyperparameter tuning
- **NEW** :
  - Multiple epochs of minibatch updates (PPO)
  - Clipped probability ratios (why?)
    - Forms a pessimistic estimate (lower-bound) of performance
  - ATARI:
    - Much better than A2C and similar to ACER (though simpler)

## **REINFORCE**

1. Initialize random policy (gaussian noise weights?)
2. Collect trajectory using these
2. Total (sum) reward of each trajectory
3. Compute estimate of the reward gradient
    - ![pimg1]
2. Update the policy using gradient ascent with learning rate alpha.
    - ![pimg2]
2. REPEAT!

## **problems?**

- **Inefficient** - We run a policy, update once, then toss trajectory
- **Noisy** - gradient estimate _g_, possibly the trajectory may not be representative of the policy.
- **No Clear Credit Assignment** - A trajectory has good/bad actions, but all receive same reward.
  - Must run more simulations to smooth out corrections
  - Signals accumulate over time

## **Solutions!**

- **Noise Reduction** - Averaging rewards
  - Take one trajectory and compute gradient - bad
- **Collect in Parallel**
  - Then average across all the multiple trajectories
    - ![pimg3]

- **Rewards Normalization** - norm the distribution
  - Multiple trajectories feature a distribution of rewards that we can use to normalize on.
  - What rewards may be good early on (2) may not be good near the end.
    - ![pimg4]
    
- **Credit Assignment**
  - All current rewards are from the past.
  - Therefore, we can discard them, as they have no effect on future actions.
  - We can just use only the future reward as the coefficient.
    - ![pimg5]
- This will help properly assign credit to the current action.

 - **Issues?**
    - It turns out mathematically, ignoring past rewards might change the current trajectory gradient, but does not change the **averaged gradient**.
    - And actually:
        - The resulting gradient is less noisy, so using future reward should speed training up!

- **Importance Sampling**
  - Recycle old trajectories instead of throwing them away.
  - ![pimg6]
  - Add a re-weighting factor ^^ in addition to just averaging.
  - ![pimg7]

- **Re-weighting the Policy Gradient**
  - The re-weighting factor is just the product of all the policy across each step.
  - ![pimg8]
  - ![pimg9]

- **The Surrogate Function**
  - We can call this approximate form of the gradient as a new object
  - Use this new gradient to perform Gradient Ascent.
    - So that we can maximize the surrogate function.
  - **BUT** : Make sure the OLD and NEW policies don&#39;t diverge too much
    - Or else the approximations can become invalid

![pimg10]



- **Clipped Surrogate Function**
  - Keep policies from diverging too much (keep ratio around 1)

![pimg11]


- **Additional Notes**
  - Normalize future rewards over all parallel agents (speed)
  - Simpler networks may perform better
    - Steadily decrease size of the nodes at each layer
  - Remember to save policy after training

