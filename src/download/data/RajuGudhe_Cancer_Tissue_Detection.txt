
# Cancer_Tissue_Detection

This is the PyTorch implementation of Dense Siamese Network for the detection of cancer tissues in histopathology scans. The architecture used in the work is shown in the figure below:
<p align="center">
 <img src="./images/proposed_model.png" alt="Drawing" width="80%">
</p>
 
 The main objective of this work is to implement  an end-to-end pipeline for deep semantic learning model for detection of cancerous tissues.

# Datasets

## CIFAR-10
The CIFAR-10 dataset is used in this work to train the model from scratch and use the trained weights to retrain model on Pcam dataset. PyTorch provide helper functions for data loading, for reference please visit this [link](https://pytorch.org/docs/stable/torchvision/datasets.html).

## PatchCamelyon(PCam)


The PatchCamelyon(PCam) is a new benchmark dataset for medical image classification. It consists of 3,27,680 color images of size 96 × 96 pixels patches extracted from histopathologic scans of lymph node sections. Each image is annotated with a binary label indicating the presence of tumor cells.



<p align="center">
 <img src="./images/pcam.png" alt="Drawing" width="50%">
</p>

The data is provided under the CCO License, Data download  [link](https://drive.google.com/drive/folders/1gHou49cA1s5vua2V5L98Lt8TiWA3FrKB). The dataset is available in HDF5 files with train, valid, test split. Each set contains the data and target file. In this work the data is preprocessed into PyTorch ImageFolder format and the structure of data folder is as follows:
```bash
├── PCam-data
	├── train
	│   ├── tumor
	│   ├── no-tumor
        ├── valid
	│   ├── tumor
	│   ├── no-tumor
	├── test
	    ├── tumor
	    ├── no-tumor

```


## Training
All experiments demand high computation power, I trained all the experiments using FloydHub(easy to configure, but need to spend a dollar for one hour training. It's better to use gpu after making sure the code runs properly) and Google Colab(most of the time running out of memory) GPU. For FloydHub, I used powerup package which includes TeslaK80 GPU. 

## Experiments

In learning process to implement the planned Dense Siamese Network, I implemented DenseNets and Siamese Networks separated and trained them  on CIFAR-10, PCam datasets.  Please refer to experiments section in this repository to run the models. Separate documentation is provided with instructions on how to train these networks. 

Few screenshots  results from the implemented DenseNet architecture on CIFAR-10 dataset:
<p align="center">
 <img src="./images/p1.png" alt="Drawing" width="30%">
  <img src="./images/p2.png" alt="Drawing" width="30%">
</p>

<p align="center">
 <img src="./images/p3.png" alt="Drawing" width="30%">
 <img src="./images/p4.png" alt="Drawing" width="30%">
</p>


## To do List

- [ ] Fix bugs to train Siamese network for PCam dataset
- [ ] Upload code for the Dense Siamese Network
- [ ] Update results for Siamese Network
- [ ] Transfer Learning 


## Citations
Biblatex entry:
```bash
# PCam dataset
@ARTICLE{Veeling2018-qh,
  title         = "Rotation Equivariant {CNNs} for Digital Pathology",
  author        = "Veeling, Bastiaan S and Linmans, Jasper and Winkens, Jim and
                   Cohen, Taco and Welling, Max",
  month         =  jun,
  year          =  2018,
  archivePrefix = "arXiv",
  primaryClass  = "cs.CV",
  eprint        = "1806.03962"
}

# DenseNets
@article{densenets,
  author    = {Gao Huang and
               Zhuang Liu and
               Kilian Q. Weinberger},
  title     = {Densely Connected Convolutional Networks},
  journal   = {CoRR},
  volume    = {abs/1608.06993},
  year      = {2016},
  url       = {http://arxiv.org/abs/1608.06993},
  archivePrefix = {arXiv},
  eprint    = {1608.06993},
  timestamp = {Mon, 10 Sep 2018 15:49:32 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/HuangLW16a},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
@INPROCEEDINGS{siamese:,
    author = {Yaniv Taigman and Ming Yang and Lior Wolf},
    title = {L.: Deepface: Closing the gap to human-level performance in face verification},
    booktitle = {In: IEEE CVPR},
    year = {2014}
}
@article{cifar,
title= {CIFAR-10 (Canadian Institute for Advanced Research)},
journal= {},
author= {Alex Krizhevsky and Vinod Nair and Geoffrey Hinton},
year= {},
url= {http://www.cs.toronto.edu/~kriz/cifar.html},
keywords= {Dataset},
terms= {}
}
```


# Acknowledgements
- Data image folder is created based on [this](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) PyTorch tutorial.
- Data creation was inspired from [this](https://medium.com/@meghana97g/classification-of-tumor-tissue-using-deep-learning-fastai-77252ae16045) blog.


