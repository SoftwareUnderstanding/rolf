
Data Augmentation using Domain Adaptation of Synthetic Data for Semantic Segmentation
===================================

_Autonomous driving is an open ongoing research topic, which has seen a tremendous rise over the last few years. We explore semantic segmentation of the cityscape dataset with additional data augmented using domain adaptation with the help of CycleGANs._


_**Contributors:** Atman Patel, Mudit Jain, Taruj Goyal, and Harshita Krishna (Team: sCVngers)_

Project Organization
------------
```
├── CycleGAN
│   ├── data
│   │   ├── data_loader.py
│   │   └── test_images
│   ├── model
│   │   ├── cycle_gan.py
│   │   ├── networks.py
│   │   ├── params.yaml
│   │   └── saved_models
│   └── scripts
│       ├── test_cycle_gan.py
│       ├── train_cycle_gan.py
│       └── utils.py
├── Demo.ipynb
├── directory_structure
├── LICENSE
├── OCNet
│   ├── data
│   │   └── dataset
│   ├── LICENSE
│   ├── model
│   │   ├── config
│   │   ├── network
│   │   └── oc_module
│   ├── output
│   │   ├── checkpoint
│   │   ├── log
│   │   └── visualize
│   └── scripts
│       ├── _config.yml
│       ├── eval.py
│       ├── generate_submit.py
│       ├── inplace_abn
│       ├── inplace_abn_03
│       ├── run_resnet101_baseline.sh
│       ├── train.py
│       └── utils
├── README.md
└── requirements.txt
```
Requirements
-----------
 - Python 3.6+
 - Git
 - PyTorch 0.4.1
- Linux (tested on Ubuntu 18.04)
- NVIDIA GPU is strongly recommended
- [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn)
 - Conda
 - Docker 19.03

To clone the repository
``` bash
$ git clone https://github.com/tarujg/domain-adapt.git
```
 ``` bash
$ pip install -r requirements.txt
```

 - NVIDIA Docker Setup (Ubuntu 16.04/18.04, Debian Jessie/Stretch/Buster)(https://hub.docker.com/r/rainbowsecret/pytorch04/tags/)
#### Add the package repositories
``` bash
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

$ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
$ sudo systemctl restart docker
```

#### Running Demo
1. Download cityspaces data to OCNet/data/data
2. Download trained model from [link](https://drive.google.com/open?id=1Kwk6yLK57ZbRk90fTwg9o1MH0Par1-v8i) to OCNet/output/checkpoint
3. Update the file OCNet/data/dataset/list/cityscapes/demo.lst as per your requirement.
4. Right now the OCNet/scripts/run_resnet101_baseline.sh is modified for testing purposes. NOTE : Same can be used for training after updating the paths and uncommenting the code
5. Run OCNet/scripts/run_resnet101_baseline.sh from domain_adapt directory
6. The semantic segmented outputs are generated in OCNet/output/visualize
7. Run the jupyter notebook from domain_adapt directory
8. The outputs for the GAN are generated in CycleGAN/data/my_saved_images

---
## Datasets
We are currently using [Cityscapes](https://www.cityscapes-dataset.com/), GTA2Cityscapes and [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/) dataset for our project.

| Cityscapes | GTA5 | GTA2Cityscapes
|:------:|:------:|:------:|
|[link](https://www.cityscapes-dataset.com/)|[link](https://download.visinf.tu-darmstadt.de/data/from_games/)|[link](http://efrosgans.eecs.berkeley.edu/cyclegta/cityscapes2gta.zip)

#### References and used code sources
- [InplaceABN](https://github.com/mapillary/inplace_abn)
- [Non-local_pytorch](https://github.com/AlexHex7/Non-local_pytorch).
- [Pytorch-Deeplab](https://github.com/speedinghzl/Pytorch-Deeplab)
- [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)
- [semantic-segmentation-pytorch](https://github.com/CSAILVision/semantic-segmentation-pytorch)
```
Citations:

1. Yuhui Yuan, Jingdong Wang and Microsoft Research. “OCNet: Object Context Network for Scene Parsing”. In: 2018
https://arxiv.org/pdf/1809.00916.pdf

2. Zhu, Jun-Yan, Taesung Park, Phillip Isola, and Alexei A. Efros. "Unpaired image-to-image translation using cycle-consistent adversarial networks." In Proceedings of the IEEE international conference on computer vision, pp. 2223-2232. 2017.
https://arxiv.org/pdf/1703.10593.pdf

```

#### Please contact us if you have any questions.
- Atman Patel at <a2patel@ucsd.edu>
- Taruj Goyal at <tgoyal@ucsd.edu>
- Mudit Jain at <mujain@ucsd.edu>
- Harshita Krishna at <h1krishn@ucsd.edu>
