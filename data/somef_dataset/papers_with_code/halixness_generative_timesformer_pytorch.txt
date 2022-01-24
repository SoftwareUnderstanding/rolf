<img src="./diagram.png" width="600px"></img>

## Generative TimeSformer - Pytorch

| :exclamation:  This repository is based on [lucidrains/TimeSformer-pytorch](https://github.com/lucidrains/TimeSformer-pytorch)   |
|-----------------------------------------|

This fork attemps to extend the purpose of TimeSformer to video generation for future frame prediction. 
The purpose is solely educative and experimental.

### The Idea
From the principle underlying in TimeSformer:
- video frames are split into patches
- for each patch, time attention is computed correspondingly to the same patch in the other frames (or timesteps)
- for each patch, space attention is computed correspondingly to the other patches in the same frame

The original TimeSformer outputs a single classification token, which attends all keys and values when attention is computed.
Thus, the idea is to define `N` tokens equal to `frames * patches_per_frame`, which are going to attend all keys and values when attention is computed; a final embedding layer is added to project these tokens to final visual patches, symmetrically to the input of the model.
Futher experiments will determine whether this solution scales to generate videos.

## Experiments
- [ ] Training on moving MNIST
- [ ] Training on KTH Actions
- [ ] Training on a custom video dataset

## TimeSformer - Pytorch

Implementation of <a href="https://arxiv.org/abs/2102.05095">TimeSformer</a>, from Facebook AI. A pure and simple attention-based solution for reaching SOTA on video classification. This repository will only house the best performing variant, 'Divided Space-Time Attention', which is nothing more than attention along the time axis before the spatial.

<a href="https://ai.facebook.com/blog/timesformer-a-new-architecture-for-video-understanding/">Press release</a>

## Usage

```python
import torch
from timesformer_pytorch import TimeSformer

model = TimeSformer(
    dim = 512,
    image_size = 224,
    patch_size = 16,
    num_frames = 8,
    num_target_frames = 4,
    channels = 3,
    out_channels = 1,
    depth = 12,
    heads = 8,
    dim_head =  64,
    attn_dropout = 0.1,
    ff_dropout = 0.1
)

video = torch.randn(2, 8, 3, 224, 224) # (batch x frames x channels x height x width)
pred = model(video) # (2, 4, 1, 224, 224)
```

## Citations

```bibtex
@misc{bertasius2021spacetime,
    title   = {Is Space-Time Attention All You Need for Video Understanding?}, 
    author  = {Gedas Bertasius and Heng Wang and Lorenzo Torresani},
    year    = {2021},
    eprint  = {2102.05095},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```
