# SimCLR4Paint
SimCLR4Paint is experimental project that run training with illustration data.

Training is ended without any errors, but model couldn't acquire appropriate representations.  
So, this project has failed.  
Maybe, contrastive learning doesn't perform well with small batch size, so batch size 8 was not enough.  
If you wanna get good representation you expect, you have to make batch size more bigger, use gradient accumlation if your gpu is not enough for training and use small image size.  
These techniques may help your model can get representation you want.

Training data is not yet shared now.   
As most of training data comes from danbooru dataset, you can use it as training data.

## Dependency
This project may depend on following packages.

- pytorch
- torchvision
- pytorch-lightning
- tqdm
- Pillow

If you met errors because of the packages, please install missing packages.

## Usage
There are two ways to run training script.

- main.py
- training-pl.ipynb

Both script used pytorch-lightning because of its usefulness and reproducibility.

### main.py
Execute following command

```
$ python main.py --train_path path/to/train_data --valid_path path/to/valid_data 
```

Description of argument

```
--seed            default: 42             seed value
--batch_size      default: 32             batch size 
--epochs          default: 10             number of epochs
--projection_dim  default: 256            output feature size
--img_size        default: 512            input image size
--temperature     default: 0.5            hyperparameter used in loss
--train_path      default: "./data/train" path of training data
--valid_path      default: "./data/valid" path of validation data
```

### training-pl.ipynb
Launch jupyter notebook or jupyter lab and open notebooks/traing-pl.ipynb, then run all cells.

## Architecture
Model has pre-trained ResNet18 body for encoding and two layer dense nn for projecting h to z described in paper.  
Pre-traind model has trained with danbooru2018 dataset and is shared in [here](https://github.com/RF5/danbooru-pretrained/)

## References
Original SimCLR paper - [arxiv](https://arxiv.org/abs/2002.05709)

The Illustrated SimCLR - [article](https://amitness.com/2020/03/illustrated-simclr/)

Understanding SimCLR - [Medium](https://medium.com/analytics-vidhya/understanding-simclr-a-simple-framework-for-contrastive-learning-of-visual-representations-d544a9003f3c)

Other SimCLR implementation
- Spijkervet - [github](https://github.com/Spijkervet/SimCLR)
- sthalles - [github](https://github.com/sthalles/SimCLR)
