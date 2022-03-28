## Application of DNN to Steganalysis

I completed this project from scratch with Guillermo Bescos and Lucas Soffer for the course CS231N ("Convolutional Neural Networks for Visual Recognition") at Stanford University.

# About the Project

Steganography is the science of hiding a message into an ordinary file. In recent years criminals and terrorists have used steganography to hide encrypted messages within ordinary images. What seems like a simple image of a cat on the internet could actually contain a secret message or even a huge file that only a specific individual could ever retrieve. This is why law enforcement are now researching steganalysis: the study of detecting steganography.
In this project we would like to use the Kaggle\cite{kaggle} dataset called "ALASKA2 image Steganalysis" where the task is to classify images which hide an encoded message. The training set consists of 75k images and for each image, an altered version with each of 3 different  undisclosed encoding algorithms (a total of 300k images). Our goal is to use deep learning architectures such as ResNet and SRnet to classify these images. We will also consider the impact of using discrete fourier transforms instead of RGB inputs.

# Setup

The shell script setup.sh enables to  download the dataset "bash setup.sh". For the required libraries, you can set up the environment with the file environment "conda env create -f environment.yml". After downloading the dataset, you should run rename.py ( the dataloader depends on the images name). According to the parameters you defined, you can train the model with train_eval.py. 

# Files Organisation

-Arguments for the neural networks are defined in args.json. They are transformed in dictionnaries and saved with te file args.py. 
-Dataloaders are created in dataload.py. There are different options depending on if we want to put the images without alteration and with alteration in the same batch. 
-Fourier transforms of image are computed in dctexp.py and fourier.py.
-In models.py, we provide the ResNet and SRnet model.  SRnet is coded from scratched and ResNet is adapted from a pretrained version. Specific layers for SRnet are coded in layers.py
-train_eval.py is used for training and evaluating the models images whereas test.py is used for testing it with the AUC metric.

# References

@misc{alaska,
    author = {Rémi Cogranne, Quentin Giboulot, Patrick Bas},
    title = {{ALASKA2 Image Steganalysis}},
    note = {\url{https://alaska.utt.fr/#timeline}},
    year = 2020,
}

@inproceedings{Yousfi2019Alaska,
 author = {Yousfi, Yassine and Butora, Jan and Fridrich, Jessica and Giboulot, Quentin},
 title = {Breaking ALASKA: Color Separation for Steganalysis in JPEG Domain},
 booktitle = {Proceedings of the ACM Workshop on Information Hiding and Multimedia Security},
 series = {IH\&\#38;MMSec'19},
 year = {2019},
 isbn = {978-1-4503-6821-6},
 location = {Paris, France},
 pages = {138--149},
 numpages = {12},
 url = {http://doi.acm.org/10.1145/3335203.3335727},
 doi = {10.1145/3335203.3335727},
 acmid = {3335727},
 publisher = {ACM},
 address = {New York, NY, USA},
} 

@ARTICLE{paper,
author={M. {Boroumand} and M. {Chen} and J. {Fridrich}}, 
journal={IEEE Transactions on Information Forensics and Security}, 
title={Deep Residual Network for Steganalysis of Digital Images}, 
year={2019}, 
volume={14},
number={5},
pages={1181-1193},
}

@article{DBLP:journals/corr/HeZRS15,
  author    = {Kaiming He and
               Xiangyu Zhang and
               Shaoqing Ren and
               Jian Sun},
  title     = {Deep Residual Learning for Image Recognition},
  journal   = {CoRR},
  volume    = {abs/1512.03385},
  year      = {2015},
  url       = {http://arxiv.org/abs/1512.03385},
  archivePrefix = {arXiv},
  eprint    = {1512.03385},
  timestamp = {Wed, 17 Apr 2019 17:23:45 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/HeZRS15.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}