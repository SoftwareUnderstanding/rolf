Lookahead Optimizer for pytorch
===============================

Installation
------------

- Optional: Nvidia Apex

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

On start script:
```
apt update
apt -y upgrade
apt install -y git rsync
conda install -y -c conda-forge -c pytorch click tensorboardx ignite tensorboard tensorflow
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
``
