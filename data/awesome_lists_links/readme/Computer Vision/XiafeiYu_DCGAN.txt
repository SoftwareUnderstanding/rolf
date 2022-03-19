# DCGAN - tensorflow

the program implement this paper: https://arxiv.org/pdf/1511.06434.pdf

## Prerequisites
python 3.5 <br> 
tensorflow 0.12.1 <br> 
gpu or cpu <br> 

## Dataset
the dataset is celebA, you can download here: [celebA](mlab.ie.cuhk.edu.hk/projects/CelebA.html) <br> 
then put your download images into celebA folder

## Usage
To train a model <br> 
```
$ python main.py --train True <br> 
```
To test a model <br> 
```
$ python main.py --train False <br> 
```
