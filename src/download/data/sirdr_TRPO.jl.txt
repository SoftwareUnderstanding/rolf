# TRPO.jl

This repository aims to provide a Julia implementation of Trust Region Policy Optimization (TRPO)[https://arxiv.org/pdf/1502.05477.pdf] for use with the JuliaPOMDP framework[https://github.com/JuliaPOMDP].

This is currently a work in progress.

Notes:

- Implement simple policy gradient
- States are the actions (Identity mapping, reward is difference between action and state, sample states randomly)
- Plot (KL, KL relative to bound)
- hessian-vector product by finite differences (difference should be around 10^-5)
- set Value Network to 0
- Value Network is there just to reduce variance (so can just use simple linear function or 0 baseline)
- checl out Open AI baselines (for identity mapping problem) -- baselines/common/tests/envs
- check number of linesearches (shouldnt be more than 5 -- approximately)
