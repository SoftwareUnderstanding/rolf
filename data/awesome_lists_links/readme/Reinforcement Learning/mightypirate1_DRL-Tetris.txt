![DRL-Logo](assets/logo.png)

# DRL-Tetris
This repository is three things:

1. It is the open-source multiplayer tetris game [SpeedBlocks] turned into a reinforcement learning (RL) environment with a complete python front-end. The Environment is highly customizable: game-field size, block types used, action type etc. are all easily changed. It is written to function well with RL at large scale by running arbitrary numbers of tetris-games in parallel, in a simple-to-use manner.

2. A multi-processing framework for running multiple workers gathering trajectories for a trainer thread. The framework is flexible enough to facilitate many different RL algorithms. If you match the format of the template-agent provided, your algorithm should work right away with the framework.

3. A small but growing family of RL-algorithms that learns to play two-player tetris through self-play:

    **SIXten** learns the value-function thru a k-step estimation scheme, utilizing the world-model of the environment and a prioritized distributed experience replay (modelled on Schaul et al.). Toghether with the multiprocessing framework described above it's similar to Ape-X (https://arxiv.org/abs/1803.00933) but the RL algorithm itself is different.

    **SVENton** is a double[1] dueling[2] k-step DQN-agent with a novel Convolutional Neuro-Keyboard interfacing it with the environment (1: https://arxiv.org/abs/1509.06461 2: https://arxiv.org/abs/1511.06581). It too utilizes the distributed prioritized experience replay, and the multi-processing framework. It is highly experimental, but it's included so SIXten doesn't get lonely.

    **SVENton-PPO** is similar to SVENton but trains by way of PPO. Trajectories are gathered by the workers, who also compute GAE-advantages [https://arxiv.org/abs/1506.02438]. This is the recommeded default for all new-comers.

The RL-framework and RL-algorithms may be separated into different repositories at some time in the future, but for now they are one.

> NOTE: The master-branch contains some features that are experimental. If there are issues, revert to the stable branch (https://github.com/mightypirate1/DRL-Tetris/tree/stable) or the kinda_stable branch (https://github.com/mightypirate1/DRL-Tetris/tree/kinda_stable). The latter "should" be stable, but testing all features is pretty time-consuming, and I try to make headway.

The quality of code has changed as I learned. Please be lenient when looking at the remnants of old code. There will come a day when it's fixed, let's hope!

## Installation:
* Pull the repository.
* Install dependencies (see "Dependencies").
* Run: `make build`

If you want to get more experimental, you can create a `virtualenv` and install `dockerfiles/requirements.local.txt` and go nuts.
#### Dependencies:

- docker
- docker-compose (1.2.7 used on dev machine)

Make sure you make them GPU-enabled.

## Usage:

To start training, we recommend starting off from the example in experiments/sixten_base.py

To run the example project using 32 environments per worker thread, and 3 worker threads (+1 trainer thread), for 10M steps, run
```
python3 thread_train.py experiments/sventon_ppo.py --steps 10000000
```
periodically during training, weights are saved to models/project_name/weightsNNN.w. Additionally, backups are made to models/project_name/weightsLATEST.w, and the final version is saved to models/project_name/weightsFINAL.w.

To test these weights out against themselves
```
python3 eval.py path/to/weightfile.w
```
or against other weights
```
python3 eval.py path/to/weightfile1.w path/to/weightfile2.w (...) --argmax
```
Settings are saved along with the weights so that it is normally possible to make bots made with different settings, neural-nets etc. play each other. As long as the game_size setting is the same across projects, they should be compatible! See "Customization" for more details.

#### Experiment-files:
A few different experiment files are provided:

* experiments/sventon_ppo.py - Trains SVENton using PPO and a res-block type architecture".
* experiments/sventon_dqn.py - Trains SVENton as above, but using a flavour of DQN.
* experiments/sixten_base.py  - Trains SIXten.

> Hyper-parameters need to be adjusted to your hardware. For good performance, you need to balance learning-rate ("value-lr") with numbers of training epochs per batch ("n_train_epochs_per_update"). The defaults provided are tuned for the test system: i5-4690K CPU @ 3.50GHz × 4 + GeForce GTX 1080 Ti + 16GB Ram, and if your system is significantly different, you might get different results. Contact me if you need advice on this.

#### Demo weights:
The project ships with some pre-trained weights as a demo. When in the DRL-Tetris folder, try for instance
```
python3 eval.py models/demo_weights/SIXten/weightsDEMO1.w
```
to watch SIXten play.

Similarly,
```
python3 eval.py models/demo_weights/SVENton/weightsDEMO1.w
```
shows SVENton in action.

SIXten was trained using a limited piece set, so it's na unfair comparison - but
```
python3 eval.py models/demo_weights/SVENton/weightsDEMO1.w models/demo_weights/SIXten/weightsDEMO1.w --all-pieces
```
shows the two agent types duking it out!

## Customization:

The entire repository uses a settings-dictionary (the default values of which are found in tools/settings.py). To customize the environment, the agent, or the training procedure, create dictionary with settings that you pass to the relevant objects on creation. For examples of how to create such a dictionary, see the existing experiment-files in "experiments/".

For minor customization, you can just edit the settings-dictionary in the corresponding experiment-file (e.g "experiments/sixten_base.py"). To change the size of the field used, just find the game_field entry and put a new value there. Any option that is in tools/settings.py can be overridden this way. For major customization you might need to code.

> This is a design-choice I am - with the benefit of experience and hindsight - not too impressed with. My attention is finite and directed elsewhere in the project for now, as this works ok for RL-dev. If you hate it, feel free to contribute! ;)

#### Pieces:
What pieces are being used is specified in the settngs-dictionary's field "pieces". It contains a list of any subset of {0,1,2,3,4,5,6}. [0,1,2,3,4,5,6] means the full set is used. The numbers correspond to the different pieces via the aliasing (L,J,S,Z,I,T,O) <~> (0,1,2,3,4,5,6). If those letters confuse you, you might want to check out https://tetris.fandom.com/wiki/Tetromino

The pre-defined settings on the master branch plays with only the O- and the L-piece to speed up training (pieces set to [0,6]).

> Quickest way to enable all pieces is to comment out the line  that reduces it to O and L, in the experiment-file you intend to use (e.g. "experiments/sixten_base.py"):
> ```
> # "pieces" : [0,6],
> ```
> "pieces" will get the default value instead, which means all pieces are used.

#### Advanced customization:

If you wish to customizations that are not obvious how to do, just contact me and I will produce the documentation needed asap. To write your own agent and/or customize the training procedure, you will have to write code. Probably the best way to get into the code is to look at the function "thread_code" in threads/worker_thread.py where the main training loop is located.

## Known issues:
There are a lot of settings and combinations thereof that are not tested, since it would be prohibitively slow to do so. This means that if you test out the project and fiddle around, you probably will run into issues. Contact me if you need help, and if you solve any, please give me a pull-request.

* If using different environments on different settings, the last one to be instantiated will impose it's settings on the others. This is generally only a problem when evaluating models trained on different settings.
* SVENton-PPO could in theory become unbalanced in that the workers could drown the trainer in samples. This is nowhere near to occur on the test system, so there has been no need to guard against it. You would notice this by looking in the print-out flow for the line telling how long the last policy update took: "trained for 3.8436920642852783s [4707 samples]", and if each time it's printed, the numbers keep increasing, you have this problem. You need to slow down the workers to solve this then. Either put workers on CPU by setting "worker_net_on_cpu" to True, or change the number of workers.

## On the horizon:

#### Network play:
So far no official client exists for playing against the agents you train. Coming soon is a closer integration of the environmnet backend and the game itself. This will allow for an evaluation mode where an agent plays versus a human player online in the same way that two human play against each other. Stay tuned!

#### API-documentation:
The environment documentation is next on the todo-list. For now I will say that the functionality is similar conceptually to the OpenAI gym environments, and should be quite understandable from reading the code (check out the function "thread_code" in threads/worker_thread.py). Having said that, if you would like to see documentation happen faster, or if you have any question regarding this, contact me and I will happily answer.

#### Standardized environment configurations:
A few standard configurations will be decided on and made official so that they are easily and reliable recreated. Basically replacing
```
settings = {...}
env = tetris_environment(...,...,...,...,settings=settings)
```
with
```
env = environment.make("FullSize-v0")
```

As I want to maintain full flexibility w.r.t what constitutes an action-space, there are no current plans on full gym-integration, but that - as all other things - might change with time.

## Contribute!
If you want to get involved in this project and want to know what needs to be done, feel free to contact me and I will be happy to discuss!

If you find a bug, have concrete ideas for improvement, think something is lacking, or have any other suggestions, I will be glad to hear about it :-)

## Contact:
yfflan at gmail dot com

[SpeedBlocks]: <https://github.com/kroyee/SpeedBlocks>
[tensorflow]: <https://www.tensorflow.org/install/>
