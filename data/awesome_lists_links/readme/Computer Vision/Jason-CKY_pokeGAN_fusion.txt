# Pytorch implementation of DCGAN and cDCGAN on Pokemon Image Dataset
Pytorch implementation of Deep Convolutional Generative Adversarial Networks (DCGAN) [1] and its conditional variant (cDCGAN) [2] for Pokemon Dataset.

Training with cDCGAN allows for fusion of different pokemon types to generate new kinds of pokemon.

* The network architecture (number of layer, layer size and activation function etc.) of this code differs from the paper.
* Pokemon dataset used the pokemon's primary and secondary type as condition.
* Spectral Normalisation is taken from https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
* you can download
  - Pokemon image dataset: https://www.kaggle.com/kvpratama/pokemon-images-dataset
  - pokemon.csv file: https://www.kaggle.com/vishalsubbiah/pokemon-images-and-types

## Results on 128x128 Resolution
* Generate using fixed noise

<table align='center'>
<tr align='center'>
<td> DCGAN</td>
<td> cDCGAN</td>
</tr>
<tr>
<td><img src = 'pretrained\dcgan\animation.gif'> 
<td><img src = 'pretrained\cdcgan\animation.gif'>
</tr>
</table>

## Pretrained weights
* DCGAN: [GoogleDrive](https://drive.google.com/file/d/1TN_39wnvahnCFkinLUn97TshF_K4M5_t/view?usp=sharing)
    * Save the file into pretrained/dcgan/
* cDCGAN: [GoogleDrive](https://drive.google.com/file/d/1a4121sFBESjLRRex1rYNjdz6lPv4Gm6M/view?usp=sharing)
    * Save the file into pretrained/cdcgan/

## Implementation details
* DCGAN
    - Batch Size = 128
    - Crop Size = 128x128
    - Size of feature maps in generator = 64
    - Size of feature maps in discriminator = 32
    - Number of training epochs = 2000
    - Learning rate for Generator = 0.0001
    - Learning rate for Discriminator = 0.0002
    - Beta1 hyperparam for Adam optimizers (momentum) = 0.5
    - Label Smoothing (Real labels change from 1.0 -> 0.9)

* cDCGAN
    - Batch Size = 128
    - Crop Size = 128x128
    - Size of feature maps in generator = 64
    - Size of feature maps in discriminator = 32
    - Number of training epochs = 5000
    - Learning rate for Generator = 0.0001
    - Learning rate for Discriminator = 0.0002
    - Beta1 hyperparam for Adam optimizers (momentum) = 0.5
    - Spectral Normalisation for the Discriminator
    - Label Smoothing (Real labels change from 1.0 -> 0.9)

* GAN Training Tricks used (https://towardsdatascience.com/10-lessons-i-learned-training-generative-adversarial-networks-gans-for-a-year-c9071159628):
    1. Two Time-Scale Update Rule
    2. Label Smoothing
    3. Spectral Normalisation

* Data Augmentations:
    1. transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0)
    2. transforms.RandomRotation(10)
    3. transforms.RandomHorizontalFlip(p=0.5)
    4. transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

## Dependencies
```
conda env create -f environment.yml
```

## How to use

### Training model from custom image dataset
```
git clone https://github.com/Jason-CKY/pokeGAN_fusion.git
cd pokemonGAN_fusion
```
Edit settings on cfg/config.ini
```
python train_dcgan.py
python train_cdcgan.py
```

### Generating new images
```
git clone https://github.com/Jason-CKY/pokeGAN_fusion.git
cd pokemonGAN_fusion
python test_dcgan.py --weights <path to dcgan Generator weights file> -bs <batchsize> --output <path to output folder> < --grid >
python test_cdcgan.py --weights <path to dcgan Generator weights file> --primary type <primary type> --secondary_type <secondary type> -bs <batchsize> --output <path to output folder>  < --grid >
```

### Issues
Mode collapse experienced on the cDCGAN model, probably due to the little variance in each type of pokemon in the dataset.
Lowering the learning rate did not help to solve the mode collapse problem.

## Reference
[1] A. Redford, L. Metz, S. Chintala (7 Jan, 2016). "Unsupervised representation learning with deep convolutional generative adversarial networks" 

(Full paper: https://arxiv.org/pdf/1511.06434.pdf)

[2] Mirza, Mehdi, and Simon Osindero. "Conditional generative adversarial nets." arXiv preprint arXiv:1411.1784 (2014).

(Full paper: https://arxiv.org/pdf/1411.1784.pdf)

