## Glow in PyTorch

![Glow](/images/glow.jpg?raw=true "Glow")

Implementation of Glow in PyTorch. Based on the paper:

  > [Glow: Generative Flow with Invertible 1x1 Convolutions](https://arxiv.org/abs/1807.03039)\
  > Diederik P. Kingma, Prafulla Dhariwal\
  > _arXiv:1807.03039_


## Usage
Use .sh files to execute
Note that --realnvp use model RealNVP to model toy data

## Samples (K=32, L=3, C=512)

### RealNVP
#### Original Data
![Original Data](/images/data.jpg?raw=true "Original Data")
#### Latent Space
![Latent Space](/images/z.jpg?raw=true "Latent Space")
#### Inferenced Data
![Inferenced Data](/images/x.jpg?raw=true "Inferenced Data")
### Epoch 40
![Samples at Epoch 40](/images/cifar10.jpg?raw=true "Samples at Epoch 40")

## Results (K=32, L=3 C=512)
### Bits per Dimension

| Epoch | Train |
|-------|-------|
| 10    | 4.91  |
| 20    | 4.69  | 
| 30    | 4.48  |
| 40    | 4.41  |
