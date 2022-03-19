# Deep Convolutional GAN
Repository provides implementation of DCGAN model (https://arxiv.org/abs/1511.06434) in TensorFlow 2.0.  Model has been testen on MNIST, CelebA and flowers102 datasets.

## Installation
* Project has been tested on Ubuntu 18.04 with Python 3.6.8 and TensorFflow 2.0.0-alpha0
* The dependencies are Pillow, tqdm and scipy libraries, all included in setup.py
* Traning requires `dcgan` lib which can be installed via `python setup.py install` command
* To download dataset with flowers images run `bash data/flowers.sh` from the repository's root

## Structure
The repository organized as follows. `dcgan` contains model itself with the data prerpocessings setups. `data` contains scripts for dataset downloading and serves as a default directory for datasets. `scripts` contain script for the configuration processing and launching of the training procedure. `results` folder accumulate results obtained during training i.e. TenorBoard data, text logs and in-between images produced by generator network. `tests` contains basic tests for presented datasets.

## Training
Settings of the training routine gathered in configuration files. Default configs for datasets are `mnist.conf`, `celeba.conf` and `flowers.conf`. They can be used to reproduce results presented below.
Run following commands from root of the repository to start training procedures
* `python scripts/train/run_train --config scripts/mnist.conf`
* `python scripts/train/run_train --config scripts/celeba.conf`
* `python scripts/train/run_train --config scripts/flowers.conf`

Training procedure generates text logs which can be found in `results/<dataset>/logs`. Data from TensorBoard are accumulated in `results/<dataset>/tensorboard`. Generated images from generator are saved into `results/<dataset>/get_output`

After training `dcgan/utils/gifmaker.py` can be used to produce gif image from samples from generator. Example of usage can be found in main section of the file.

## Datasets
__mnist__

60,000 black and white handwritten digits of size 28x28

__CelebA__

202,599 number of face images of various celebrities with 10,177 unique identities. Images are cropped to contain faces in their centers during preprocessing stage. All images are RGB.

__flowers__

Dataset of 102 different flowers from Visual Geometry Group of Oxford university. Data was downloaded from http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html. Each image was different size, so samples were cropped and resized to be 64x64 RBG images. As original dataset consists only of 8k images data augmentation was performed. Each image was rotated by 5 different angles and each angle translated by four directions with gaussian noise on top. As a result training was launched upon 172k images.

## Tests
Tests contains training procedure that stops after very first batch, run following commands to check
* `python -m unittest tests/test_mnist.py`
* `python -m unittest tests/test_celeba.py`
* `python -m unittest tests/test_flowers.py`

## Results
Representative outputs from generator can be found in `results` folder. 
__MNIST__

![2-900](https://user-images.githubusercontent.com/23639048/56539459-a9bc2b00-656e-11e9-9c1e-b984c13264e3.jpg)
![mnist](https://user-images.githubusercontent.com/23639048/56539479-b80a4700-656e-11e9-93a4-56ca2c35350b.gif)
<img width="1064" alt="curves" src="https://user-images.githubusercontent.com/23639048/56539252-fc491780-656d-11e9-96b1-cc3363b47843.png">

__CelebA__

![celeba](https://user-images.githubusercontent.com/23639048/56539310-36b2b480-656e-11e9-926e-a11cc1b2babd.gif)

__flowers__

![flowers](https://user-images.githubusercontent.com/23639048/56539601-133c3980-656f-11e9-8184-29ebb74fb5fc.jpg)
![flowers](https://user-images.githubusercontent.com/23639048/56539347-53e78300-656e-11e9-99d6-bcafaea6f215.gif)

## References
[1] Alec Radford, Luke Metz, Soumith Chintala _Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks_ (https://arxiv.org/abs/1511.06434)

[2] Ian J. Goodfellow _Generative Adversarial Networks_ (https://arxiv.org/abs/1406.2661)
