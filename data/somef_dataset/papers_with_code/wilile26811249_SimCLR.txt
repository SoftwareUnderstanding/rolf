# SimCLR
A PyTorch implementation of SimCLR based on ICML 2020 paper [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709).

![Network Architecture image from the paper](SimCLR_arch.png)



## Usage

### Train SimCLR

```bash=
python pretext.py
```

```bash=
usage: pretext.py [-h] [--represent_dim REPRESENT_DIM]
                  [--temperature TEMPERATURE] [--k K]
                  [--batch_size BATCH_SIZE] [--epochs EPOCHS]
                  [--gpu_device GPU_DEVICE] [--seed S]

SimCLR

optional arguments:
  -h, --help            show this help message and exit
  --represent_dim REPRESENT_DIM
                        Feature dim for latent vector
  --temperature TEMPERATURE
                        Temperature used in softmax
  --k K                 Top k most similar images used to predict the label
  --batch_size BATCH_SIZE
                        Input batch size for training (default: 512)
  --epochs EPOCHS       Number of epochs to train (default: 500)
  --gpu_device GPU_DEVICE
                        Select specific GPU to run the model
  --seed S              Random seed (default: 1)
```

### Linear Evaluation
```bash=
python downstream.py
```

```bash=
usage: downstream.py [-h] [--model_path MODEL_PATH] [--batch_size BATCH_SIZE]
                     [--epochs EPOCHS] [--gpu_device GPU_DEVICE] [--seed S]

Linear Evaluation

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        The pretrained model path
  --batch_size BATCH_SIZE
                        Input batch size for training (default: 512)
  --epochs EPOCHS       Number of epochs to train (default: 100)
  --gpu_device GPU_DEVICE
                        Select specific GPU to run the model
  --seed S              Random seed (default: 1)
```

---
# Reference
```
Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. arXiv:2002.05709, 2020.
```

# Author
**Hong-Jia Chen**