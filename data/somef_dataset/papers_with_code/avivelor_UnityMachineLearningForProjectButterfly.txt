# UnityMachineLearningForProjectButterfly

Aviv Elor - aelor@ucsc.edu - avivelor1@gmail.com


It's official, we've been published! More details behind this project can be found at the following manuscript: 
[*Elor, A., & Kurniawan, S. (2020, August). Deep Reinforcement Learning in Immersive Virtual Reality Exergame for Agent Movement Guidance. In 2020 IEEE 8th International Conference on Serious Games and Applications for Health(SeGAH). IEEE, 2020.*](https://www.researchgate.net/publication/344380137_Deep_Reinforcement_Learning_in_Immersive_Virtual_Reality_Exergame_for_Agent_Movement_Guidance)


What if we could train a virtual robot arm to guide us through our physical exercises, compete with us, and test out various double-jointed movements?
This project is an exploration of Unity ML-Agents on training a double-jointed "robot arm" to protect butterflies with bubble shields in an immersive virtual environment.
The arm is trained through utilizing General Adversarial Imitation Learning (GAIL) and Reinforcement Learning through Proximal Policy Optimization (PPO) to play an Immersive Virtual Reality Physical Exercise Game.
Overall, this was a fun, deep dive into exploring Machine Learning through mlagents.
Feel free to try out the standalone build if you want to attempt to predict torque values for two joints better than a neural network, or use the VR build to compete head to head with a neural network-driven arm for the same VR exercise game!

If any questions, email aelor@ucsc.edu and or message Aviv Elor.

# Imitation Learning and Virtual Reality Gameplay

To explore imitation learning with PPO, refer to the Unity ML-Agents Documentation at https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Readme.md and https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-Imitation-Learning.md .
This section explored the application of General Adversarial Imitation Learning (GAIL) with Deep Reinforcement Learning through Proximal Policy Optimization (PPO).
Demonstrations were recorded with an HTC Vive 2018 VR System by utilizing two 2018 Vive Trackers on a human demonstrator's shoulder and elbow joint to capture torque and angular momentum using Unity's Fixed Joint API.

To get started, download Anaconda and set up a virtual environment through conda activate mlagents (see this example to configure your environment https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Readme.md).
Enter the training scene at the following:

```sh
~\UnitySDK\Assets\ML-Agents\Examples\CM202-ExoArm\Scenes\
```

To capture demonstrations in VR, place the SteamVR trackers at the elbow and shoulder joint of the human user. 
Set the Observations of the agent to be "Heuristic Only." 
Check the record demonstration box in the Demonstration Recorder script and have the human user perform ideal and concise movements.
After recording the demonstration, update the config yaml files to point to the demonstration for GAIL.

Pre-recorded demonstrations for Project Butterfly can be found at:
```sh
~\UnitySDK\Assets\Demonstrations\
```

Training configuration with GAIL for Project Butterfly can be found at:
```sh
~\config\trainer_config_exoarm.yaml
```

After demonstrations are recorded, proceed back to the training scene to begin agent learning.
With the anaconda terminal, prepare to train through using the following terminal command:

```sh
mlagents-learn config/gail_config_exoimitationarm.yaml --run-id=<run-identifier> --train --time-scale=100
```

Now sit back and let the model train. After checkpoints are saved, you can use tensorboard to examine the model's performance:

```sh
tensorboard --logdir=summaries
```

The trained model for this section can be found at:

```sh
~\models\ImitationButterfly-0\ExoReacher.nn
or
~\UnitySDK\Assets\ML-Agents\Examples\CM202-ExoArm\TFModels\ImitationReacher\
```

A demo video of this section can be found at: https://youtu.be/ckMaDXHUGrw

[![IMAGE ALT TEXT](http://img.youtube.com/vi/ckMaDXHUGrw/0.jpg)](http://www.youtube.com/watch?v=ckMaDXHUGrw "Imitation Learning Demo Video")

# Reinforcement Learning and Non-VR Based Gameplay

To mess around with deep reinforcement learning and training, refer to the Unity ML-Agents documentation at https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Readme.md.
To get started, download Anaconda and set up a virtual environment through conda activate mlagents (see this example to configure your environment https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Readme.md).
Enter the training scene at the following:

```sh
~\UnitySDK\Assets\ML-Agents\Examples\CM202-ExoArm\Scenes\
```

With the anaconda terminal, prepare to train through using the following terminal command:

```sh
mlagents-learn config/trainer_config_exoarm.yaml --run-id=<run-identifier> --train --time-scale=100
```

Now sit back and let the model train. After checkpoints are saved, you can use tensorboard to examine the model's performance:

```sh
tensorboard --logdir=summaries
```

Subsequently, I was able to train the robot arm very well through utilizing 16 agents in parallel through Deep Reinforcement Learning with Proximal Policy Optimization (PPO).
After four hours of training, my reward slowly rose from 0.1 to 40 (where +0.01 reward was given per every frame the arm successfully protected the butterfly).
See the demo video below for a discussion of the training, results, and demo experience.

The trained model for this section can be found at:

```sh
~\models\ExoReacherPBF-0\ExoReacher.nn
or
~\UnitySDK\Assets\ML-Agents\Examples\CM202-ExoArm\TFModels\ExoReacher\
```

A demo video of this section can be found at: https://youtu.be/5J7xes28bZA

[![IMAGE ALT TEXT](http://img.youtube.com/vi/5J7xes28bZA/0.jpg)](http://www.youtube.com/watch?v=5J7xes28bZA "Reinforcement Learning Demo Video")

# Materials and References

Materials:
* Virtual Reality Demo (HTC Vive, Stable) - https://github.com/avivelor/UnityMachineLearningForProjectButterfly/raw/master/UnitySDK/ExoButterflyVR-HTCViveBuild.zip
* Standalone Downloadable Demo (Stable) - https://github.com/avivelor/UnityMachineLearningForProjectButterfly/raw/master/UnitySDK/ExoButterfly-StandaloneBuild.zip
* Imitation Learning and Human vs Neural Network Research Video - https://youtu.be/ckMaDXHUGrw
* Early Reinforcement Learning Demo Video - https://youtu.be/5J7xes28bZA
* Blog Posts - https://www.avivelor.com/

External Tools Used and Modified for this Project:
* Unity Machine Learning Agents Beta - https://github.com/Unity-Technologies/ml-agents
* Project Butterfly - https://www.avivelor.com/post/project-butterfly
* Unity ML Agents Introduction - https://towardsdatascience.com/an-introduction-to-unity-ml-agents-6238452fcf4c
* Unity ML Agents Reacher Example - https://github.com/Unity-Technologies/ml-agents/tree/master/Project/Assets/ML-Agents/Examples/Reacher
* Older Unity ML Reacher Example by PHRABAL  - https://github.com/PHRABAL/DRL-Reacher
* Proximal Policy Optimization (PPO) in Unity - https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-PPO.md
* General Adversarial Imitation Learning (GAIL) in Unity - https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-Imitation-Learning.md
* Deep Reinforcement Learning (through Deep Deterministic Policy Gradient or DDPG) -  https://arxiv.org/pdf/1509.02971.pdf
* HTC Vive Virtual Reality System (2018) - https://www.vive.com/us/product/vive-virtual-reality-system/
* HTC Vive Trackers (2018) - https://www.vive.com/us/vive-tracker/

Reading References:
* Unity Machine Learning - https://unity3d.com/machine-learning
* Academic Paper on Project Butterfly at IEEEVR 2019 Paper by Elor et Al - https://www.researchgate.net/publication/335194991_Project_Butterfly_Synergizing_Immersive_Virtual_Reality_with_Actuated_Soft_Exosuit_for_Upper-Extremity_Rehabilitation
