# RegNet

Unofficial PyTorch implementation of RegNet based on paper [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678).

---

## Table of Contents
* [Model Architecture](#model-architecture)
* [RegNetX and RegNetY models](#regnetx-and-regnety-models)
* [Usage](#usage)
* [Experiments Results (ImageNet-1K)](#experiments-results-imagenet-1k)
* [Citation](#citation)


---
## Model Architecture
<figure>
<img src="resources/model_gen_arch.png" alt="Trulli" style="width:100%">
<figcaption align = "center"><b>General network structure</b></figcaption>
</figure>

<figure>
<img src="resources/model_xblock.png" alt="Trulli" style="width:100%">
<figcaption align = "center"><b>X block based on the standard residual bottleneck
block with group convolution</b></figcaption>
</figure>

---
## RegNetX and RegNetY models
<figure>
<img src="resources/model_regnetx.png" alt="Trulli" style="width:100%">
<figcaption align = "center"><b>Top RegNetX Models</b></figcaption>
</figure>

<figure>
<img src="resources/model_regnety.png" alt="Trulli" style="width:100%">
<figcaption align = "center"><b>Top RegNetY Models</b></figcaption>
</figure>

---

## Usage
### Training
- Single node with one GPU
```bash=
python main.py
```

- Single node with multi GPU
```bash=
CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.launch --nproc_per_node=2 --master_port=6666 main_ddp.py
```


```bash=
optional arguments:
  -h, --help            show this help message and exit
  --gpu_device GPU_DEVICE
                        Select specific GPU to run the model
  --batch-size N        Input batch size for training (default: 64)
  --epochs N            Number of epochs to train (default: 20)
  --num-class N         Number of classes to classify (default: 10)
  --lr LR               Learning rate (default: 0.01)
  --weight-decay WD     Weight decay (default: 1e-5)
  --model-path PATH     Path to save the model
```

---

## Experiments Results (ImageNet-1K)

![Training Accuracy](./resources/train_acc.png)

![Validation Accuracy](./resources/val_acc.png)

![Loss](./resources/loss_curve.png)


|Model  |  params(M) |   batch size | epochs | train(hr) |   Acc@1  |  Acc@5  |
|-------|:------:|:----:|:--------:|:--------:|:-------:|:-------:|
|REGNETY-400MF | 4.4  | 256 | 90 |  39   |  71.522%  |  90.146% |

---

## Citation
```
@InProceedings{Radosavovic2020,
  title = {Designing Network Design Spaces},
  author = {Ilija Radosavovic and Raj Prateek Kosaraju and Ross Girshick and Kaiming He and Piotr Doll{\'a}r},
  booktitle = {CVPR},
  year = {2020}
}
```


### If this implement have any problem please let me know, thank you.