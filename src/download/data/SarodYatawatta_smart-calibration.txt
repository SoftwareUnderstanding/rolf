# Smart Calibration
Using reinforcement learning for hyperparameter tuning in calibration of radio telescopes, and in other data processing pipelines (like elastic net regression). Code to accompany the paper [Deep reinforcement learning for smart calibration of radio telescopes](https://academic.oup.com/mnras/advance-article-abstract/doi/10.1093/mnras/stab1401/6276731) [(preprint)](https://arxiv.org/abs/2102.03200).

RL agent code is based on [this code](https://github.com/philtabor/Youtube-Code-Repository.git).

Implemented in PyTorch, using openai.gym. Algorithms tested are: [DDPG](https://arxiv.org/abs/1509.02971),  [TD3](https://arxiv.org/abs/1802.09477) and [SAC](https://arxiv.org/abs/1801.01290). The figure below shows the performance of the three.

<img src="figures/comparison.png" alt="Performance of the algorithms learing the elastic net problem" width="400"/>

## Elastic net regression

<img src="figures/enet_pipeline.png" alt="Elastic net regression agent and environment" width="700"/>

Run ``` main_{ddpg|td3|sac}.py ``` to use DDPG or TD3 or SAC.

Files included are:

``` autograd_tools.py ```: utilities to calculate Jacobian, inverse Hessian-vector product etc.

``` enetenv.py ```: openai.gym environment

``` enet_td3.py  ```:  TD3 agent

``` enet_ddpg.py ```: DDPG agent

``` enet_sac.py ```: SAC agent

``` enet_eval.py ```: evaluation and compare with sklearn GridSearchCV

``` lbfgsnew.py ```: LBFGS optimizer

``` main_ddpg.py ```: run this for DDPG

``` main_td3.py ```: run this for TD3

``` main_sac.py ```: run this for SAC

## Calibration

Additional packages: python-casacore, astropy,  openmpi

Simulation and Calibration software: [SAGECal](https://github.com/nlesc-dirac/sagecal)

Imaging software: [Excon](https://sourceforge.net/projects/exconimager/)


### Influence maps
Influence maps give a visual representation of the [influence function](https://academic.oup.com/mnras/article/486/4/5646/5484901) of radio interferometric calibration. Here is a sample:

<img src="figures/influence_maps.png" alt="Influence maps" width="700"/>

We use influence maps as part of the state representation. An analogy to this is visualizing an untrained and a trained CNN model. The figures below show the influence function for CIFAR10 classifier using AlexNet (untrained), AlexNet (trained) and ResNet18 (trained). With training, the figures become less random, but still has non-zero features implying overfitting in the training.

<img src="figures/alexnet_untrained.png" alt="Alexnet untrained" width="300"/>

<img src="figures/alexnet_trained.png" alt="Alexnet trained" width="300"/>

<img src="figures/resnet18_trained.png" alt="ResNet18 trained" width="300"/>

Note the difference between the trained AlexNet (65% accuracy) and ResNet18 (80% accuracy), the latter has much less bias.

Run ``` main_{ddpg|td3|sac}.py ``` to use DDPG or TD3 or SAC.
You can copy some example data from [here](https://github.com/nlesc-dirac/sagecal/tree/master/test/Calibration).

The figure below shows small areas in the sky before (left) and after (right) calibration.

<img src="figures/calibration.png" alt="Before and after calibration" width="400"/>

Files included are:

``` calibration_tools.py ```: utility routines

``` simulate.py ```: generated sky models, systematic errors for data simulation 

``` calibenv.py ```: openai.gym environment

``` analysis.py ```: calculate influence function/map

``` calib_td3.py ```: TD3 agent

``` calib_ddpg.py ```: DDPG agent

``` calib_sac.py ```: SAC agent

``` lbfgsnew.py ```: LBFGS optimizer

``` main_ddpg.py ```: run this for DDPG

``` main_td3.py ```: run this for TD3

``` main_sac.py ```: run this for SAC

``` docal.sh ```: shell wrapper to run calibration

``` doinfluence.sh ```: shell wrapper to run influence mapping

``` dosimul.sh  ```: shell wrapper to run simulations

``` inspect_replaybuffer.py ```: inspect the replay buffer contents

The following scripts are for handling radio astronomical (MS) data

``` addcol.py ```: add new column to write data

``` changefreq.py ```: change observing frequency

``` readcorr.py ```: read data and output as text

``` writecorr.py ```: write text input and write to MS

``` addnoise.py  ```: add AWGN to data

``` calmean.sh ```: calculate mean image

``` doall.sh ```: wrapper to do simulation, calibration, and influence map generation
