# optimization
Intro to numerical optimization method: papers recommendation and code implementation.
The following contents can be found in [keras-optimizers](https://keras.io/optimizers/#sgd)
### examples
| Date | codes |
| ---- | ---- |
| Oct 10, 2018 | [Non-negative matrix factorization](https://github.com/suzyi/optimization/blob/master/notebook/nonneg_matrix_fact.ipynb) |
## 1 - Theory
Deep learning models are typically trained by a stochastic gradient descent optimizer. There are many variations of stochastic gradient descent: Adam, RMSProp, Adagrad, etc. For an understanding intro to these algorithm, see [here](https://papers.nips.cc/paper/19-optimization-with-artificial-neural-network-systems-a-mapping-principle-and-a-comparison-to-gradient-based-methods.pdf). All of them let you set the learning rate. This parameter tells the optimizer how far to move the weights in the direction of the gradient for a mini-batch. If the learning rate is low, then training is more reliable, but optimization will take a lot of time because steps towards the minimum of the loss function are tiny. If the learning rate is high, then training may not converge or even diverge. Weight changes can be so big that the optimizer overshoots the minimum and makes the loss worse. The training should start from a relatively large learning rate because, in the beginning, random weights are far from optimal, and then the learning rate can decrease during training to allow more fine-grained weight updates.
## general optimzation method
### SGD
+ paper - ""
### RMSprop
+ [Slides](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) by Geoffrey Hinton.
### Nestrov
### Adagrad
+ [Paper-2011](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) published on Journal of Machine Learning Research, cited by 3178 (It's August 5, 2018.).
### Adadelta
+ [Paper-2012](https://arxiv.org/abs/1212.5701) on Arxiv, cited by 1960 (It's August 5, 2018.).
### Adam
+ [Adam-2014](https://arxiv.org/abs/1412.6980) and [its convergence-2018](https://openreview.net/forum?id=ryQu7f-RZ), have been cited by 11125 and 31, respectively.
### Adamax
### Nadam
## convex optimization method
Here we list several mature optimization provided by cvxpy, solvers such as ECOS, OSQP, SCS and their link can be found [here](https://www.cvxpy.org/tutorial/advanced/index.html). Especially, SCS can be used to solve non-negative matrix factorization problem.
+ ADMM
### problems that are well solved in convex framework
#### Second-oder cone programming (SOCP)
+ Problems that can be trasferred as SOCP. For more details, see Boyd's paper "Applications of second-Order cone
programming".
### 1 - 1 - convex optimization
### 1 - 2 - non-convex optimization
+ paper `Two-Player Games for Efficient Non-Convex Constrained Optimization`
+ The convex-concave procedure [(CCCP)](https://papers.nips.cc/paper/2125-the-concave-convex-procedure-cccp.pdf)
## 2 - code and programming
+ How to numerically compute derivatives?
## 3 - Optimization framework
### 3 - 1 - CVXOPT
+ [CVXOPT](http://cvxopt.org/) is a free software package for convex optimization based on the Python programming language, developed by Martin Andersen, Joachim Dahl, and Lieven Vandenberghe.
+ [variables](http://cvxopt.org/userguide/modeling.html#variables)- It seems like variables can only be vector variable, from official guide.
### 3 - 2 - CVXPY
#### intro to cvxpy
+ [CVXPY](http://www.cvxpy.org/) is a Python-embedded modeling language for convex optimization problems, Steven Diamond, Eric Chu, Akshay Agrawal and Stephen Boyd.
+ [Github for cvxpy](https://github.com/cvxgrp/cvxpy)
+ [Variables](https://www.cvxpy.org/tutorial/intro/index.html) can be scalars, vectors, or matrices, meaning they are 0, 1, or 2 dimensional.
+ [Library of examples](http://www.cvxpy.org/examples/index.html)
+ must follow [DCP rules](http://cvxr.com/cvx/doc/dcp.html).
+ [Geometric programming](https://github.com/cvxgrp/cvxpy/issues/32). We'll add log_sum_exp soon and then you'll be able to write GP's in their convex formulation. But there won't be a GP mode for the foreseeable future (answered in 2014).
#### installation
+ Step 1 - You'd better have cvxpy installed in a virtual environment.
+ Step 2 - You need to have `numpy` and `scipy` in stalled in your virtual environment in advance.
+ Step 3 - `pip install cvxpy`
### 3 - 3 - CVX
+ [CVX](http://cvxr.com/cvx/) is a Matlab-based modeling system for convex optimization, developed by Michael Grant and Stephen Boyd.
+ [Variables](http://cvxr.com/cvx/doc/basics.html) can be real or complex scalars, vectors, matrices, or n-dimensional arrays.
+ must follow [DCP rules](http://cvxr.com/cvx/doc/dcp.html).
+ [Geometric programmings](http://cvxr.com/cvx/doc/gp.html) are special mathematical programs that can be converted to convex form using a change of variables. The convex form of GPs can be expressed as DCPs, but CVX also provides a special mode that allows a GP to be specified in its native form. CVX will automatically perform the necessary conversion, compute a numerical solution, and translate the results back to the original problem.
### 3 - 4 - openopt
+ [Openopt](http://openopt.org) solves general (smooth and nonsmooth) nonlinear programs, including problems with integer constraints. Unlike CVXOPT, it has no software for solving semidefinite programs. The solvers were all written by Dmitrey Kroshko himself and don't have a long history, so testing was probably limited. OpenOpt itself does _not- interface to general third party solvers.
+ [Github for openopt](https://github.com/troyshu/openopt)
### Gurobi
+ [Gurobi](http://www.gurobi.com/) is an optimization software. According to [wikipedia](https://en.wikipedia.org/wiki/Gurobi), it supports both python and matlab mainly for solving Linear Programming (LP), Quadratic Programming (QP), Quadratically constrained Programming (QCP), mixed integer linear programming (MILP), mixed-integer quadratic programming (MIQP), and mixed-integer quadratically constrained programming (MIQCP).
+ [yalmip](https://yalmip.github.io/)
## examples
### an constrained (simple linear constraints) optimization problem
This problem [(notebook)](https://github.com/suzyi/optimization/blob/master/notebook/constrainedOpt.ipynb) appears at the inverse process of a relu DNN.
### try to write a package like `Autograd`, using python
this package should have the function of automatic differentiation.
## Distributed Optimization
Distributed Optimization is also known as decentralized optimization.

Paper recommendations:
+ Optimal Algorithms for Non-Smooth Distributed Optimization in Networks. Kevin Scaman, Francis Bach. And references therein.
+ Distributed optimization and statistical learning via the alternating direction method of multipliers. Stephen Boyd, Neal Parikh, Eric Chu and etc.
