# Pointer networks

Use pointer network to compute the convex hull. 

This is an unofficial implementation of the research paper: https://arxiv.org/abs/1506.03134


<img src="imgs/network.png" alt="pointer network" width="300"/>


## Training

- Create a dataset: `python3 data.py`
- Training: `python3 run_training.py`


## Inference

- Run: `python3 predict_convex_hull.py -c ckpts/hk_state_0099999.ckpt -d 256 -l 200 -s 42`


## Attention hidden subspace

+ Checkout `01-pca-attention.ipynb` for the details.

+ Encoder attention hidden states are in a 2d plane which is a 1-to-1 mapping from the point space.

<img src="imgs/encoder_attention_pca.jpg" alt="encoder" width="600"/>

+ Decoder attention hidden states go around a circle in the 2d hidden plane.

<img src="imgs/original_att.jpg" alt="decoder" width="400"/>
