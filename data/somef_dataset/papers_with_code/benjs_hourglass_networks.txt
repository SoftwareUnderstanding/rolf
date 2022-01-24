![Tests](https://github.com/benjs/hourglass_networks/actions/workflows/test.yml/badge.svg)
![PyPI package](https://github.com/benjs/hourglass_networks/actions/workflows/pypi.yml/badge.svg)

# tf.keras Implementation of Hourglass Models for Human Pose Estimation

Implementation of the paper [Toward fast and accurate human pose estimation via soft-gated skip connections](https://arxiv.org/abs/2002.11098) by Bulat et al. and [Stacked Hourglass Networks for Human Pose Estimation](https://arxiv.org/abs/1603.06937) by Newell et al. using tensorflow keras.

This work has been done as part of a student course at [Intelligent Process Automation and Robotics Lab (IPR)](https://www.ipr.kit.edu/english/index.php).

## Project setup
Initial requirements:
  - Python 3.8+
  - [Tensorflow GPU software requirements](https://www.tensorflow.org/install/gpu#software_requirements) for TF>=2.5.0

### Through pip
```bash
pip install tf-hourglass-networks
```

### From source
```bash
git clone https://github.com/benjs/hourglass_networks.git hg-nets
cd hg-nets

python3 -m venv venv/
source venv/bin/activate

pip install -r requirements.txt
```

## Training
Start training with
```
python -m hourglass.training --config default_config.yaml --visible-devices 0
```
or in short
```
python -m hourglass.training -c default_config.yaml -v 0
```

Most parameters are adjusted in the config file and some can be changed through passing args. 
See `python -m hourglass.training --help` for more information.

### Data setup
Download the mpii dataset images from their [official site](http://human-pose.mpi-inf.mpg.de/#download)
and the converted annotations `train.h5` and `valid.h5` by Newell et al. from the [stacked hourglass repository](https://github.com/princeton-vl/pose-hg-train/tree/master/data/mpii/annot).
Specify the path to both parent directories in the config file.
```yaml
# Paths are relative to home
mpii_annot_dir: 'mpii_annot'
mpii_image_dir: 'mpii_images'
```

### Logging
The training process can be viewed either through tensorboard or [Weights and Biases](wandb.ai).
Adjust the following line in your config file.
```yaml
logger: 'wandb'  # or 'tensorboard'
```

### Config overwriting
Add 
```
overwrite_config: default_config.yaml
```

to a config file to take all parameters from another config file and update them with the
parameters from the current file. 
The files have to be in the same directory.

## Unit tests
Run all unit tests with
```
python -m pytest
```
