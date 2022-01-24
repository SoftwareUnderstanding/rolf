# PyTorch SSD

[日本語](https://github.com/birosjh/pytorch_ssd/blob/main/README_JP.md)

This is a PyTorch implementation of the Single Shot Multibox Detector: https://arxiv.org/pdf/1512.02325.pdf

The objective of this project is to help me get better acquainted with the components of SSD and how they all tie together.  In addition, I would like to understand what is written in the research paper, and how it might differ from common implementations.

My implementation was heavily influenced by these implementations:

- https://github.com/NVIDIA/DeepLearningExamples/tree/49e387c788d606f9711d3071961c35092da7ac8c/PyTorch/Detection/SSD
- https://github.com/amdegroot/ssd.pytorch
- https://github.com/kuangliu/pytorch-ssd/blob/master/encoder.py

For training, I used the VOC Dataset 2017 version: http://host.robots.ox.ac.uk/pascal/VOC/

### Environment

This project uses a PyTorch docker container as an environment.
https://hub.docker.com/r/pytorch/pytorch

1. Build the container

```bash
docker build . -t ssd
```

2. Start the container

a. Locally
```bash
docker run -itd --name ssd -v $(pwd):/app ssd
```

b. On Server
```bash
docker run -p 6006:6006 --gpus all -itd --name ssd -v .:/app ssd
```

3. SSH into the container
```bash
docker exec -it ssd bash
```