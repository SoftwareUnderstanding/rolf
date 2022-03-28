# FCOS: Fully Convolutional One-Stage Object Detection

The full paper is available at: [https://arxiv.org/abs/1904.01355](https://arxiv.org/abs/1904.01355). 


**It performs well on DOTA data set**


## Required hardware
We use 4 Nvidia p100 GPUs. \  

## Installation
#### Testing-only installation 
For users who only want to use FCOS as an object detector in their projects, they can install it by pip. To do so, run:
```
pip install torch  # install pytorch if you do not have it
pip install git+https://github.com/tianzhi0549/FCOS.git
# run this command line for a demo 
fcos https://github.com/tianzhi0549/FCOS/raw/master/demo/images/COCO_val2014_000000000885.jpg
```
Please check out [here](fcos/bin/fcos) for the interface usage.

#### For a complete installation 
This FCOS implementation is based on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). Therefore the installation is the same as original maskrcnn-benchmark.

Please check [INSTALL.md](INSTALL.md) for installation instructions.
You may also want to see the original [README.md](MASKRCNN_README.md) of maskrcnn-benchmark.

## Models
For your convenience, we provide the following trained models (more models are coming soon).

**ResNe(x)ts:**

*All ResNe(x)t based models are trained with 16 images in a mini-batch and frozen batch normalization (i.e., consistent with models in [maskrcnn_benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)).*

Model | AP (minival) |
--- |:---:|
FCOS_imprv_R_101_FPN_2x | 73.93 | 
#Other tests to be tested

### If you want to train with your own data
This project use the json annotation file with COCO format.
Make your directory layout like this:
```
.
└── trainset
    ├── images
    │   ├── 1.png
    │   └── 2.png
    └── labelTxt
        ├── 1.txt
        └── 2.txt
```
A example of the \*.txt files ('1' means the object is difficult):
```
x1 y1 x2 y2 x3 y3 x4 y4 plane 0
x1 y1 x2 y2 x3 y3 x4 y4 harbor 1
```
Run the following Python snippet, and it will generate the json annotation file:
```python
from txt2json import collect_unaug_dataset, convert
img_dic = collect_unaug_dataset( os.path.join( "trainset", "labelTxt" ) )
convert( img_dic, "trainset",  os.path.join( "trainset", "train.json" ) )
```

### If you want to reproduce the results on DOTA

Config: `configs/glide/dota.yaml`

#### 1. Prepare the data

Clone DOTA_Devkit as a sub-module:

```shell
REPO_ROOT$ git submodule update --init --recursive
REPO_ROOT/fcos_core/DOTA_devkit$ sudo apt-get install swig
REPO_ROOT/fcos_core/DOTA_devkit$ swig -c++ -python polyiou.i
REPO_ROOT/fcos_core/DOTA_devkit$ python setup.py build_ext --inplace
```
Edit the `config.json` and run:

```shell
REPO_ROOT$ python prepare.py
```

## 2 Training
    REPO_ROOT$ python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS tools/train_net.py --config-file $PATH_TO_CONFIG   
## 3  Test
    REPO_ROOT$ python tools/test_net.py --config-file configs/glide/dota.yaml --ckpt model
## Contributing to the project
Any pull requests or issues are welcome.

## Citations
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.
```
@inproceedings{tian2019fcos,
  title   =  {{FCOS}: Fully Convolutional One-Stage Object Detection},
  author  =  {Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  booktitle =  {Proc. Int. Conf. Computer Vision (ICCV)},
  year    =  {2019}
}
```

