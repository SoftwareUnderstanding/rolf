# InfoGAN Pytorch Implementation
This is my implementation of paper [InfoGAN](https://arxiv.org/abs/1606.03657).

## DEV Environment
- OS: Ubuntu 16.04
- GPU: Nvidia GTX 1080Ti
- Nvidia GPU Driver: 418.87.01
- CUDA: 10.1
- Python: 3.6.8
- Pytorch: 1.3.0
- torchvision==0.4.1

## Available Data
- MNIST
  
## Usage
```bash
@ project root
pip install -r requirements.txt

# with local visdom server
python -m visdom.server
python src/main.py --use_visdom True

# with remote visdom server
## In remote machine
python -m visdom.server
## In local machine
python src/main.py --use_visdom True --visdom_server <http://your_server_ip>

# See training process in http://localhost:8097 in the machine where your visdom server is running

# without visdom logger
python src/main.py --use_visdom False

```

## Directory structure
```
.
├── data
│   └── mnist
│       └── MNIST
├── README.md
├── requirements.txt
├── results
│   └── Vanila_InfoGAN2542 # results of trained model
│       ├── checkpoint
│       ├── config.json
│       ├── gifs
│       └── images
└── src
    ├── config.py
    ├── data_loader.py
    ├── __init__.py
    ├── main.py
    ├── models
    │   ├── __init__.py
    │   └── mnist
    ├── trainer.py
    └── utils.py
```
## Results

### MNIST

#### Settings

   - Latent Variables (Dim=74)
        - 1 Discrete Latent Code ~ Cat(k=10, p=0.1), Dimension = 10
        - 2 Continuous Latent Code ~ Unif(-1,1), Dimension = 2
        - 62 Random latent ~ N(0,1), Dimension = 62
   - Optimizer = Adam
        - beta1 = 0.5
        - beta2 = 0.999
   - Weight(Lambda) for latent code
        - Discrete : 1
        - Continuous : 0.1
   - Learning Rate
        - Generator: 0.001
        - Discriminator: 0.0002
   - Batch size = 128
   - Continuous Latent code is optimized by minimizing negative log-likelihood function of Gaussian distribution.

#### Training process visualization
##### Loss of Generator
![Loss of Generator](./examples/loss_G.png)

##### Loss of Discriminator
![Loss of Discriminator](./examples/loss_D.png)

##### Probability from Discriminator for real/fake samples
![Probability from Discriminator for real/fake samples](./examples/prob_D.png)

##### Information Loss
![Information Loss](./examples/loss_Info.png)

##### Loss from discrete variable
![Loss from discrete variable](./examples/loss_discrete.png)

##### Losses from each continuous variables
![Losses from each continuous variables](./examples/loss_cont.png)

#### Results of latent traverse
- x-axis : change of a continuous latent variable in linspace(start=-1, end=1, steps=10) while fix other continuous variable to 0
- y-axis : each category of discrete variable from 0~9

##### Latent Traversal for continuous latent code index = 1
Representing Rotation

![Latent Traversal for continuous latent code index = 1](./examples/latent_traversal_c_1.png)

##### Latent Traversal for continuous latent code index = 2
Representing Width

![Latent Traversal for continuous latent code index = 2](./examples/latent_traversal_c_2.png)

#### Animated results of fixed latent setting during training

##### Animated results from training for 20 epochs (Discrete code index = 1, Continuous code index =1)
![Animated results from training for 20 epochs (Discrete code index = 1, Continuous code index = 1)](./examples/GIF-Cd_1-Cc_1.gif)

##### Animated results from training for 20 epochs (Discrete code index = 1, Continuous code index = 2)
![Animated results from training for 20 epochs (Discrete code index = 1, Continuous code index = 2)](./examples/GIF-Cd_1-Cc_2.gif)
## References

### Code
I referenced existing pytorch based Infogan implementations.
- https://github.com/pianomania/infoGAN-pytorch
- https://github.com/Natsu6767/InfoGAN-PyTorch
- https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/infogan
- https://github.com/taeoh-kim/Pytorch_InfoGAN

### Theory
- https://www.inference.vc/infogan-variational-bound-on-mutual-information-twice/
- https://arxiv.org/abs/1606.03657

## ToDo

- [x] Implement Vanila InfoGAN
- [x] Add visdom visualizer
- [x] Add animator to produce gif results
- [x] Successful reproduce of Infogan for MNIST dataset
- [ ] Add dSprite dataset for measuring disentanglement performance of InfoGAN
- [ ] Implement Disentanglement metrics