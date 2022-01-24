# Caffe-SGDR: Stochastic Gradient Descent with Restarts

Caffe implementation of SGDR from "SGDR: Stochastic Gradient Descent with Restarts" by Ilya Loshchilov and Frank Hutter (http://arxiv.org/abs/1608.03983)

![lr_cosine_annealing](img/lr_cosine_annealing.png)


## Usage

### Installation

1. Copy the code file in `src/` dir to your own caffe src `${CAFFE_ROOT}/src/caffe/solvers/sgd_solver.cpp`

2. Add 2 additional variables in `SolverParameter` in `${CAFFE_ROOT}/src/caffe/proto/caffe.proto`
```
message SolverParameter {
  ...
  // mult_Factor, min_lr used in sgdr policy
  optional int32 mult_factor = 43 [default = 1];
  optional float min_lr = 44 [default = 0];
}
```

Update the variables `id` (i.e., 43, 44 in the above code) if necessary.

### Solver Setting

|  Variable   | Meaning or Setting |
| :---------: | :----------------: |
|  lr_policy  |       "sgdr"       |
|   base_lr   |    $\eta_{max}$    |
|   min_lr    |    $\eta_{min}$    |
|  stepsize   |       $T_0$        |
| mult_factor |     $T_{mult}$     |

## Some Results on Image Caption Task

### NIC

[
Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555)

![nic_performance](img/nic.png)

### Up-down Captioner

[
Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://arxiv.org/abs/1707.07998)

![updown_performance](img/updown.png)