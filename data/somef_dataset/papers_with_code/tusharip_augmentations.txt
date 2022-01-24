# Augmentations
Augmentations: a Technique to increase the diversity of your training set by applying random (but realistic) transformations.

*This repo contains, augmentations related to vision, Text, Audio*

***
## Vision
***
Augmentations : Color Jitter, Cutout, Mixup, Cutmix

***
### [*Color jitter*]()

 ColorJitter is a type of image data augmentation where we randomly change the brightness, contrast and saturation of an image.

1. [**Brightness**]() : refers to the overall lightness or darkness of the image
    <p align="center"
    <br>
    <br>
    <img src="./outputs/vision/brightness.jpg" width="300" height ="150">
    </p>
2. [**Contrast**]() : refers to the difference in luminance or colour that makes an object distinguishable
    <p align="center"
    <br>
    <br>
    <img src="./outputs/vision/contrast.jpg" width="300" height ="150">
    </p>
3. [**Saturation**]() : refers to  the intensity and purity of a color as displayed in an image. The higher the saturation of a color, the more vivid and intense it is. The lower a color's saturation, the closer it is to pure gray on the grayscale
    <p align="center"
    <br>
    <br>
    <img src="./outputs/vision/saturation.jpg" width="300" height ="150">
    </p
***
### [*CutOut*]()    
 Cutout augmentation is a kind of regional dropout strategy in which a random patch from an image is zeroed out (replaced with black pixels). Cutout samples suffer from the decrease in information and regularization capability.

<p align="center">
  <img width="300" height="150" src="./outputs/vision/cutout.png">
</p>

### [*Mixup*](https://arxiv.org/pdf/1710.09412.pdf)
In Mixup augmentation two samples are mixed together by linear interpolation of their images and labels. Mixup samples suffer from unrealistic output and ambiguity among the labels and hence cannot perform well on tasks like image localization and object detection.
<p align="center">
  <img width="300" height="150" src="./outputs/vision/mixup.png">
</p>

### [*CutMix*](https://github.com/clovaai/CutMix-PyTorch)
In CutMix augmentation we cut and paste random patches between the training images. The ground truth labels are mixed in proportion to the area of patches in the images. CutMix increases localization ability by making the model to focus on less discriminative parts of the object being classified and hence is also well suited for tasks like object detection.

<p align="center">
  <img width="300" height="150" src="./outputs/vision/cutmix.png">
</p>


<p align="center"
    <br>
    <br>
    <img src="./outputs/vision/cut.jpg" width="500" height ="200">
    </p>




## Audio
Dft

## Text
## Citations
```bibtex
@misc{grill2020bootstrap,
      title={Bootstrap your own latent: A new approach to self-supervised Learning}, 
      author={Jean-Bastien Grill and Florian Strub and Florent Altché and Corentin Tallec and Pierre H. Richemond and Elena Buchatskaya and Carl Doersch and Bernardo Avila Pires and Zhaohan Daniel Guo and Mohammad Gheshlaghi Azar and Bilal Piot and Koray Kavukcuoglu and Rémi Munos and Michal Valko},
      year={2020},
      eprint={2006.07733},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

```bibtex
@misc{yun2019cutmix,
      title={CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features}, 
      author={Sangdoo Yun and Dongyoon Han and Seong Joon Oh and Sanghyuk Chun and Junsuk Choe and Youngjoon Yoo},
      year={2019},
      eprint={1905.04899},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bibtex
@misc{zhang2018mixup,
      title={mixup: Beyond Empirical Risk Minimization}, 
      author={Hongyi Zhang and Moustapha Cisse and Yann N. Dauphin and David Lopez-Paz},
      year={2018},
      eprint={1710.09412},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
T
