# Towards Learning of Filter-Level Heterogeneous Compression of Convolutional Neural Networks

Code repository for Quantized NAS (Chapter 3): https://arxiv.org/abs/1904.09872

## Installation
We recommend using virtual environment.
Installing instructions can be found in the following link: https://www.tensorflow.org/install/pip

After the virtual environment activation, we have to install the required packages:
```
pip install -r requirements.txt
```
Make sure the current directory is the repository main directory.

## Datasets
We worked with CIFAR-10 and CIFAR-100.

Both can be automatically downloaded by torchvision.

## Usage

### Search
To carry out quantized search, use the following command:
```
PYTHONPATH=../ CUDA_VISIBLE_DEVICES=0 python3 ./train_search.py --data ../data/ --dataset cifar10 --batch_size 250 --arch_learning_rate 0.1 --learning_rate 0.01 --lmbda 1 --bitwidth 2#2,4#3#8 --baselineBits 3 --epochs 1 --model thin_resnet --nCopies 1 --grad_estimator layer_same_path --alphas_regime alphas_weights_loop --nSamples 3 --workers 2 --train_portion 0.5  --gpu 0 --alphas_data_parts 4 --pre_trained "../pre_trained/cifar10/train_portion_1.0/[(32, 32)],[thin_resnet]/model.updated_stats.pth.tar"
```
Make sure the current directory is the **cnn** directory.

### Checkpoint evaluation
During the search, we sample configurations from the current distribution.
Use the following command in order to train the sampled configurations and evaluate their quality.
```
PYTHONPATH=../ CUDA_VISIBLE_DEVICES=0 python3 ./train_opt2.py --data ../data/ --json results/checkpoints/20190501-121257-1-4.json
```
Make sure the current directory is the **cnn** directory.

The argument --json holds the path to the checkpoint we would like to train.

## Acknowledgments  
The research was funded by ERC StG RAPID.  
  
## Citation  
If our work helped you in your research, please consider cite us.  
```
@ARTICLE{2019arXiv190409872Z,
       author = {{Zur}, Yochai and {Baskin}, Chaim and {Zheltonozhskii}, Evgenii and
         {Chmiel}, Brian and {Evron}, Itay and {Bronstein}, Alex M. and
         {Mendelson}, Avi},
        title = "{Towards Learning of Filter-Level Heterogeneous Compression of Convolutional Neural Networks}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computer Vision and Pattern Recognition, Computer Science - Machine Learning, Computer Science - Neural and Evolutionary Computing},
         year = "2019",
        month = "Apr",
          eid = {arXiv:1904.09872},
        pages = {arXiv:1904.09872},
archivePrefix = {arXiv},
       eprint = {1904.09872},
 primaryClass = {cs.CV},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019arXiv190409872Z},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
    
This work is licensed under the Creative Commons Attribution-NonCommercial  
4.0 International License. To view a copy of this license, visit  
[http://creativecommons.org/licenses/by-nc/4.0/](http://creativecommons.org/licenses/by-nc/4.0/) or send a letter to  
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.