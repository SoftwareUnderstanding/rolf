# SimCLR
Implementation of SimCLR (A Simple Framework for Contrastive Learning of Visual Representations)


* **Requirements**
  * numpy
  * torch
  * torchvision
  * opencv-python
  
* **command**
```bash
  - python3 main.py --epochs [epochs] --batch_size [B] --temperature [T] --strength [S] --out_dim [out_dim] --num_worker [N] --valid_size [val_size] 
  - python3 linear_eval.py --batch_size [B] --simclr_path [path] --dataset [dataset] --hid_dim [hid_dim] --num_worker [N] --finetune --baseline
```
  
* **results**

|Dataset|STL10|CIFAR10|
|------|---|---|
|Baseline|54.801|70.653|
|No Finetune|66.192|50.401|
|Finetune|73.866|71.795|

### Reference
1. A Simple Framework for Contrastive Learning of Visual Representations (https://arxiv.org/abs/2002.05709)
2. STL-10 Dataset (https://cs.stanford.edu/~acoates/stl10/)
