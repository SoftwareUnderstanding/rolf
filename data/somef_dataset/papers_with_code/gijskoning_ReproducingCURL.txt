# Reproducing CURL 
##### By Gijs Koning and Chiel de Vries
This repository houses a reproduction of [CURL](https://arxiv.org/pdf/2004.04136.pdf). This is a neural network that aims to learn a useful representation of image data to be used for reinforcement learning. 

## Motivation 
Reinforcement learning is a promising area in the field of machine learning. It is important for the future of Robotics and industrial automation. Therefore, we would like to learn more about this subject by trying to reproduce this paper. Furthermore, CURL is an unsupervised network, meaning that the network learns without the use of a ground truth. We believe that unsupervised learning is an very important topic because of this. Labelled data is expensive and time consuming to acquire. If a neural net can learn its task without the use of labelled data, it is much cheaper to run. 

The project is part of the "Seminar Computer Vision by Deep Learning" course (CS4245) at TU Delft. Our work is relevant for this course because it concerns the creation of a useful representation of image data. This is a classic computer vision task that uses state of the art neural nets to accomplish its goal.

Finally, we want to push ourselves by documenting the process thoroughly. We think this can help us greatly in our studies and teach us not only about deep learning, but also about ourselves.

## Relevant Pages
- [Project Plan](docs/project_plan.md)
- [Log](docs/log.md)
- [Agenda](docs/agenda.md)
- [Installation](docs/installation.md)
- [Encoders](docs/encoders.md)

## Other related papers and information
- [Soft Actor Critic (SAC)](https://arxiv.org/abs/1801.01290) (Haarnoja et al., 2018)   
  [Documentation](https://spinningup.openai.com/en/latest/algorithms/sac.html)  
  [Implementation used by paper](https://github.com/denisyarats/pytorch_sac_ae) (Yarats et al., 2019): 
- Reinforcement Learning with Augmented Data
- Learning Invariant Representations for Reinforcement Learning without Reconstruction
- Decoupling Representation Learning from Reinforcement Learning
- data-efficient reinforcement learning with self-predictive representations
- [CURL github](https://github.com/MishaLaskin/curl)  
  [CURL for atari](https://github.com/aravindsrinivas/curl_rainbow)


### Start training
- `conda env create -f conda_env.yml`
- `conda activate curl`
- With CURL encoder: `bash scripts/run.sh`
- Without encoder: `bash scripts/run_identity.sh`
- Visualize training (for cartpole): `tensorboard --logdir tmp/cartpole --host localhost  --reload_interval 30 --host 0.0.0.0`