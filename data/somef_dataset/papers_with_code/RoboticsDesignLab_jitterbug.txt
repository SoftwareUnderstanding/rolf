# jitterbug-dmc

A 'Jitterbug' under-actuated continuous control Reinforcement Learning domain,
implemented using the [MuJoCo](http://mujoco.org/) physics engine and
distributed as an extension to the
[Deep Mind Control suite (`dm_control`)](https://github.com/deepmind/dm_control).
This model is also available on the
[MuJoCo forum resources page](http://www.mujoco.org/forum/index.php?resources/the-jitterbug-domain.29/).

![Jitterbug model](figures/jitterbug.jpg)

## Installation

This package is not distributed on PyPI - you will have to install it from
source:

```bash
$> git clone github.com/aaronsnoswell/jitterbug-dmc
$> cd jitterbug-dmc
$> pip install .
```

To test the installation:

```bash
$> cd ~
$> python
>>> import jitterbug_dmc
>>> jitterbug_dmc.demo()
```

## Requirements

This package is designed for Python 3.6+ (but may also work with Python 3.5) 
under Windows, Mac or Linux.

The only pre-requisite package is
[`dm_control`](https://github.com/deepmind/dm_control).

## Usage

### DeepMind control Interface

Upon importing `jitterbug_dmc`, the domain and tasks are added to the standard
[`dm_control`](https://github.com/deepmind/dm_control) suite.
For example, the `move_from_origin` task can be instantiated as follows;

```python
from dm_control import suite
from dm_control import viewer
import jitterbug_dmc
import numpy as np

env = suite.load(
    domain_name="jitterbug",
    task_name="move_from_origin",
    visualize_reward=True
)
action_spec = env.action_spec()

# Define a uniform random policy
def random_policy(time_step):
    return np.random.uniform(
        low=action_spec.minimum,
        high=action_spec.maximum,
        size=action_spec.shape
    )

# Launch the viewer
viewer.launch(env, policy=random_policy)
```

### OpenAI Gym Interface

For convenience, we also provide an [OpenAI Gym](https://gym.openai.com/docs/)
compatible interface to this environment using the
[`dm2gym`](https://github.com/zuoxingdong/dm2gym) library.

```python
from dm_control import suite
import jitterbug_dmc

env = JitterbugGymEnv(
    suite.load(
        domain_name="jitterbug",
        task_name="move_from_origin",
        visualize_reward=True
    )
)

# Test the gym interface
env.reset()
for t in range(1000):
    observation, reward, done, info = env.step(
        env.action_space.sample()
    )
    env.render()
env.close()
```

### Heuristic Policies

We provide a heuristic reference policy for each task in the module
[`jitterbug_dmc.heuristic_policies`](jitterbug_dmc/heuristic_policies.py). 

### Tasks

This Reinforcement Learning domain contains several distinct tasks.
All tasks require the jitterbug to remain upright at all times.

 - `move_from_origin` (easy): The jitterbug must move away from the origin
 - `face_direction` (easy): The jitterbug must rotate to face a certain
   direction
 - `move_in_direction` (easy): The jitterbug must achieve a positive velocity
   in a certain direction
 - `move_to_position` (hard): The jitterbug must move to a certain cartesian
   position 
 - `move_to_pose` (hard): The jitterbug must move to a certain cartesian
   position and face in a certain direction 
   
### RL Algorithms

Four algorithms are implemented in `benchmark.py`, all using the
[`stable-baselines`](https://github.com/hill-a/stable-baselines) package:

 - DDPG
 - PPO2
 - SAC
 - TD3

To start a SAC agent learning the `move_in_direction` task, enter the
following command from the 'benchmarks' directory:
`python benchmark.py --alg sac --task move_in_direction --logdir /path/to/desired/directory/`.
The learning performances of the 4 algorithms on each task is shown in
[`manuscript/figures/fig-rl-perf.pdf`](manuscript/figures/fig-rl-perf.pdf).
This figure can also be generated using
[`fig-rl-perf.ipynb`](fig-rl-perf.ipynb). 

A list of hyper-parameters can be found in the Excel file
[`benchmarks/rl-hyper-params.xlsx`](benchmarks/rl-hyper-params.xlsx).
This table also gives examples of hyper-parameters derived from
[`rl-zoo`](https://github.com/araffin/rl-baselines-zoo).

### Autoencoders

Several types of autoencoders can be found in [`benchmarks`](benchmarks):
 
 - A classic AutoEncoder (AE) in
   [`benchmarks/autoencoder.py`](benchmarks/autoencoder.py)
 - A Denoising AutoEncoder (DAE) in
   [`benchmarks/denoising_autoencoder.py`](benchmarks/denoising_autoencoder.py)
 - A Dynamic Denoising AutoEncoder (DDAE) in
   [`benchmarks/ddae.py`](benchmarks/ddae.py)
 - A Variational AutoEncoder (VAE) in
   [`benchmarks/VAE.py`](benchmarks/VAE.py) (paper [arXiv:1312.6114](https://arxiv.org/abs/1312.6114))
 - A Linear Latent Dynamic Variational AutoEncoder (LLD VAE) in
   [`benchmarks/VAE_LLD.py`](benchmarks/VAE_LLD.py) (paper
   [arXiv:1506.07365](https://arxiv.org/abs/1506.07365))
 
 After training an autoencoder, it can be used by setting one of these
 Jitterbug attributes to True, depending on the autoencoder to use:
 `self.use_autoencoder`, `self.use_denoising_autoencoder`,
 `self.use_VAE`, `self.use_VAE_LLD`.
 Note that the name of the file containing the autoencoder model needs to be
 specified in the `self.jitterbug_autoencder.load_autoencoder()` function.

### Augmented Sequential Learning

To make the learning process more robust, `benchmark.py` offers the
possibility to learn sequentially using augmented Jitterbugs.
An augmented Jitterbug is a randomly modified version of the original XML file. 
To sequentially run 10 simulations with different randomly shaped Jitterbugs,
enter the command
`python benchmark.py --alg sac --task move_in_direction --logdir /path/to/desired/directory/ --domain augmented_jitterbug --num_sim 10`.
From this, it will execute the following algorithm:
 
 - Step 1: Generate an `augmented_jitterbug.xml` file by randomly modifying
   the original `jitterbug.xml` file.
 - Step 2: Start learning a policy for 1e6 steps.
 - Step 3: Save the policy and go back to step 1. Repeat the process 10 times.
 
The results of such a sequential learning are shown in figure
[`manuscript/figures/sac10seq.pdf`](manuscript/figures/sac10seq.pdf).
 
Note that by default, only the shape of the legs and the mass are modified.
More features can be tweaked such as (see
[`jitterbug_dmc/augmented_jitterbug.py`](jitterbug_dmc/augmented_jitterbug.py):
 
 - CoreBody1 density
 - CoreBody2 density
 - The global density
 - The gear
 
Examples of augmented Jitterbugs are displayed below:

Augmented Jitterbug #1
![Augmented Jitterbug #1](figures/aj1.png)

Augmented Jitterbug #2
![Augmented Jitterbug #2](figures/aj2.png)

## Common Problems

### Ubuntu: Problems with GLFW drivers 

If you're using Ubuntu 16.04, you may have problems with the GLFW dirvers.
Switching to osmesa (software rendering) may fix this,

```bash
export MUJOCO_GL=osmesa
```

### OpenMPI Wheel Fails To Build



### `libprotobuf` Version Mismatch Error

We observed this happening sometimes on Ubuntu 16.04.5 LTS when running
`import jitterbug_dmc` from python, even when the installed version of protobuf
is correct.
It seems to be something wrong with the Ubuntu tensorflow build that gets
installed by pip.
However, this doesn't seem to stop the `benchmarks/benchmark.py` file from
working.

```bash
[libprotobuf FATAL google/protobuf/stubs/common.cc:61] This program requires version 3.7.0 of the Protocol Buffer runtime library, but the installed version is 2.6.1.  Please update your library.  If you compiled the program yourself, make sure that your headers are from the same version of Protocol Buffers as your link-time library.  (Version verification failed in "bazel-out/k8-opt/genfiles/tensorflow/core/framework/tensor_shape.pb.cc".)
terminate called after throwing an instance of 'google::protobuf::FatalException'
  what():  This program requires version 3.7.0 of the Protocol Buffer runtime library, but the installed version is 2.6.1.  Please update your library.  If you compiled the program yourself, make sure that your headers are from the same version of Protocol Buffers as your link-time library.  (Version verification failed in "bazel-out/k8-opt/genfiles/tensorflow/core/framework/tensor_shape.pb.cc".)
```

Some links to more information;

 - https://devtalk.nvidia.com/default/topic/1037736/jetson-tx2/protobuf-version-error/
 - https://devtalk.nvidia.com/default/topic/1046492/tensorrt/extremely-long-time-to-load-trt-optimized-frozen-tf-graphs/post/5315675/#5315675
 - https://askubuntu.com/questions/1029394/protobuf-error-on-ubuntu-16-using-tensorflow
 - https://devtalk.nvidia.com/default/topic/1008180/tensorflow-and-protobuf-/
 - https://github.com/NVIDIA/DIGITS/issues/2061
 - https://stackoverflow.com/questions/46627874/protobuf-version-mismatch#50481381
