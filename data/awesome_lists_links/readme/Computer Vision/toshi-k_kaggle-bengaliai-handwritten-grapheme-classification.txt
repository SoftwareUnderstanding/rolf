Bengali.AI Handwritten Grapheme Classification
==============================================

I implemented Mixup and Cutmix for this Kaggle competition ["Bengali.AI Handwritten Grapheme Classification"](https://www.kaggle.com/c/bengaliai-cv19), 693rd place.

## Software

* CUDA
    * CUDA 10.2
    * CUDNN 7.6.0.3
* Pathon
    * Python 3.6.3
    * numpy==1.14.0
    * pandas==0.25.3
    * matplotlib==3.1.2
    * opencv-python==4.1.1.26
    * torch==1.4.0
    * pytorch-ignite==0.2.1
    * albumentations==0.4.3
    * tqdm==4.42.1

# Acknowledgement

I used Iafoss's functions for preprocessing.

- Iafoss / Image preprocessing (128x128)<br>https://www.kaggle.com/iafoss/image-preprocessing-128x128

I used below GeM layer for global pooling.
- filipradenovic / cnnimageretrieval-pytorch<br>https://github.com/filipradenovic/cnnimageretrieval-pytorch

# References

- Fine-tuning CNN Image Retrieval with No Human Annotation<br>Filip Radenović, Giorgos Tolias, Ondřej Chum<br>https://arxiv.org/abs/1711.02512
- mixup: Beyond Empirical Risk Minimization<br>Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz<br>https://arxiv.org/abs/1710.09412
- CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features<br>Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, Youngjoon Yoo<br>https://arxiv.org/abs/1905.04899
