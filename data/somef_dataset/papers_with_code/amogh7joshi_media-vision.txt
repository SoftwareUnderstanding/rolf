# MediaVision

A toolbox for reconstructing visual media, primarily images and videos, through operations including
colorizing, upscaling, interpolating, and more. This project makes use of entirely open-source papers and 
code implementations, which are re-implemented, and the neural networks have pre-trained weights loaded in 
(the toolbox is implemented almost exclusively in PyTorch).

**Note**: While the goal of MediaVision is to try to reconstruct media, all features that are added to images
or video, such as artificial coloring or frame interpolation (adding artificial frames), is not necessarily 
historically accurate, rather it simply *aims to provide a plausible historical interpretation*.

## Usage

### Install from Source

Currently, MediaVision is only available as a toolkit downloadable from source (though both a Colab implementation
and a pip package are in development). To install, it, first clone the repository:

```shell
git clone https://github.com/amogh7joshi/media-vision.git
```

Then, you will need to download the trained weights files for certain modules which have their 
networks with pretrained weights in Google Drive. Download them from the following list:

1. **Interpolation** (RIFE): Follow the instructions at [https://github.com/hzwer/arXiv2020-RIFE](https://github.com/hzwer/arXiv2020-RIFE).
2. **Upscaling** (ESRGAN): Follow the instructions at [https://github.com/xinntao/ESRGAN](https://github.com/xinntao/ESRGAN).

All of the pretrained weights should be placed in the `mediavision.models` directory. After you have 
done this, enter the top-level directory and execute the following: 

```shell
make build
```
 
The provided Makefile will configure the `models` directory as well as the `mediavision.weights` module
for easy usage, in addition to installing all system requirements.

## Core Features

MediaVision contains a number of modules for visual media processing and reconstruction.

### Colorization

**API**: `mediavision.colorize()`

Creates a colorized version of a grayscale image, trying to emulate traditional colors and 
emphasize vibrancy and realism. In essence, the input image will be colorized based on the neural network's
understanding of modern and historical colors.

Currently, the existing module makes use of the [Colorful Image Colorization](https://arxiv.org/abs/1603.08511) 
approach, with a direct feed-forward CNN (which contains some branches), and the code is sourced from the 
[official implementation](https://github.com/richzhang/colorization).

### Interpolation

**API**: `mediavision.interpolate()`

Performs Video Frame Interpolation (VFI) to increase a video's FPS by generating intermediate
frames in between the existing ones to increase fluidity and smoothness.

Currently, the existing module makes use of the [RIFE](https://arxiv.org/abs/2011.06294) approach, with
intermediate flow estimation and three semi-sequential models, and the code is sourced from the 
[official implementation](https://github.com/hzwer/arXiv2020-RIFE).

### Upscaling

**API**: `mediavision.upscale()`

Upscales images by enlarging them to a greater resolution while preventing quality loss, attempting to
generate realistic textures maintain visual quality.

Currently, the existing module makes use of the [ESRGAN](https://arxiv.org/abs/1809.00219) approach,
using Residual-in-Residual Dense Blocks and using features before activation, and the code is sourced from
the [official implementation](https://github.com/xinntao/ESRGAN).

## Additional Features

### Image/Video Visualization

**API**: `imagevision.visualize`

A collection of image and video visualization methods to facilitate easy viewing of core processing 
results and also aid in debugging, in certain cases. All the visualizations are constructed using
either some form of matplotlib or OpenCV backend.


## References

### Media Colorization

```bibtex
@inproceedings{zhang2016colorful,
  title={Colorful Image Colorization},
  author={Zhang, Richard and Isola, Phillip and Efros, Alexei A},
  booktitle={ECCV},
  year={2016}
}

@article{zhang2017real,
  title={Real-Time User-Guided Image Colorization with Learned Deep Priors},
  author={Zhang, Richard and Zhu, Jun-Yan and Isola, 
          Phillip and Geng, Xinyang and Lin, Angela S and Yu, 
          Tianhe and Efros, Alexei A},
  journal={ACM Transactions on Graphics (TOG)},
  volume={9},
  number={4},
  year={2017},
  publisher={ACM}
}
```

### Video Frame Interpolation

```bibtex
@article{huang2020rife,
  title={RIFE: Real-Time Intermediate Flow Estimation 
               for Video Frame Interpolation},
  author={Huang, Zhewei and Zhang, Tianyuan and Heng, 
          Wen and Shi, Boxin and Zhou, Shuchang},
  journal={arXiv preprint arXiv:2011.06294},
  year={2020}
}
```

### Media Resolution Upscaling

```bibtex
@InProceedings{wang2018esrgan,
    author = {Wang, Xintao and Yu, Ke and Wu, Shixiang and Gu, Jinjin and Liu, Yihao and Dong, Chao and Qiao, Yu and Loy, Chen Change},
    title = {ESRGAN: Enhanced super-resolution generative adversarial networks},
    booktitle = {The European Conference on Computer Vision Workshops (ECCVW)},
    month = {September},
    year = {2018}
}
```









