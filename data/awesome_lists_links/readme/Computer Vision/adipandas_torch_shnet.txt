# torch_shnet

Well documented Pytorch implementation of **Stacked-Hourglass Network (shnet)** for human pose estimation.


### Installation

Install the requirements using the following commands in your Python Environment:

```bash
pip install PyYAML
pip install h5py
pip install numpy
pip install opencv-contrib-python
pip install imageio 
pip3 install torch torchvision torchaudio
pip install pytorch-lightning
```

### MPII dataset
Refer **[this](data/MPII/README.md)** for downloading MPII dataset.

### Training

Recommended to use multi-gpu training. I haven't tested ``train.py`` which is using ``DistributedDataParallel``.

(For single GPU, reduce the ``batch_size`` in **[config.yaml](config.yaml)** to ``4``. But this may lead to convergence issues.)

To start training:
``python train_pl.py``

### References:
1. Newell, Alejandro, Kaiyu Yang, and Jia Deng. "Stacked hourglass networks for human pose estimation." European conference on computer vision. Springer, Cham, 2016. [[arxiv](https://arxiv.org/abs/1603.06937)]
2. **Stacked Hourglass Network** model implementation was adopted from **Chris Rockwell**'s implementation available in **[this GitHub repository](https://github.com/princeton-vl/pytorch_stacked_hourglass)**.
