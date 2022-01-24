# Image segmentation 

### Preferred
- Anaconda Python distribution
- PyCharm

## Getting Started

- Create environment and install requirements
- Clone this repository

```bash
git clone https://github.com/tbullmann/imagesegmentation-tensorflow.git
```

- Create directories

```bash
mkdir datasets  
mkdir temp 
```

- Download Drosophila VNC dataset
```bash
git clone https://github.com/tbullmann/groundtruth-drosophila-vnc datasets/vnc
```

- Download Mouse Cortex dataset
```bash
git clone https://github.com/tbullmann/groundtruth-mouse-cortex datasets/cortex
```

- [Run the examples](docs/README.md)

## TODO

Major issue
- Evaluation of the labelling of EM images
    - Survey on network type
    - Survey on loss
    - Survey on network depth (number of layers)
    - Survey on amount of training
    - **New** Survey on filter size and depth (number of features)

Minor issues
- For prediction and text read an image from input path and determine width, height 
- Support for 1-channel (gray scale) png as input and output
- flexible learning rate for the Adams solver

Future features
- one hot coding for labels from index colors (e.g. up to 256 categories in gif images)

## DONE

- Prediction directly from images to labels (with same filenames)
- Keeping only classic losses, classic networks 

## Acknowledgement

This repository is based on [this](https://github.com/affinelayer/pix2pix-tensorflow) Tensorflow implementation of the paired image-to-image translation ([Isola et al., 2016](https://arxiv.org/pdf/1611.07004v1.pdf)) 
Highway and dense net were adapted from the implementation exemplified in [this blog entry](https://chatbotslife.com/resnets-highwaynets-and-densenets-oh-my-9bb15918ee32). 

## Citation

If you use this code for your research, please cite the papers this code is based on:


@inproceedings{johnson2016perceptual,
  title={Perceptual losses for real-time style transfer and super-resolution},
  author={Johnson, Justin and Alahi, Alexandre and Fei-Fei, Li},
  booktitle={European Conference on Computer Vision},
  pages={694--711},
  year={2016},
  organization={Springer}
}

@article{He2016identity,
  title={Identity Mappings in Deep Residual Networks},
  author={Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
  journal={arXiv preprint arXiv:1603.05027},
  year={2016}}

@article{Srivastava2015highway,
  title={Highway Networks},
  author={Rupesh Kumar Srivastava and Klaus Greff and J{\"{u}}rgen Schmidhuber},
  journal={arXiv preprint arXiv:1505.00387},
  year={2015}
}

@article{Huang2016dense,
  title={Densely Connected Convolutional Networks},
  author={Gao Huang and Zhuang Liu and Kilian Q. Weinberger},
  journal={arXiv preprint arXiv:1608.06993},
  year={2016}
}

```