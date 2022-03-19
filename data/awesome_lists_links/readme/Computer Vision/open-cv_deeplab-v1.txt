## DeepLab v2

### New release

DeepLab v2 has been released recently (see [this](https://bitbucket.org/aquariusjay/deeplab-public-ver2)), which attains 79.7% on the challenging PASCAL VOC 2012 image segmentation task.

DeepLab v2 also incorportates some of the key layers from our DeepLab v1 (this repository). Note that there are still some minor differences between argmax and softmax_loss layers for DeepLabv1 and v2. If you want to reproduce our ICCV'15 results, please refer to the implementation of DeepLabv1.

Please also see our project [website](http://liangchiehchen.com/projects/DeepLab.html) for details.

=================================================================================
## DeepLab v1 (this repository)

### Introduction

DeepLab is a state-of-art deep learning system for semantic image segmentation built on top of [Caffe](http://caffe.berkeleyvision.org).

It combines densely-computed deep convolutional neural network (CNN) responses with densely connected conditional random fields (CRF).

This distribution provides a publicly available implementation for the key model ingredients first reported in an [arXiv paper](http://arxiv.org/abs/1412.7062), accepted in revised form as conference publication to the ICLR-2015 conference. 
It also contains implementations for methods supporting model learning using only weakly labeled examples, described in a second follow-up [arXiv paper](http://arxiv.org/abs/1502.02734).
Please consult and consider citing the following papers:

    @inproceedings{chen14semantic,
      title={Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs},
      author={Liang-Chieh Chen and George Papandreou and Iasonas Kokkinos and Kevin Murphy and Alan L Yuille},
      booktitle={ICLR},
      url={http://arxiv.org/abs/1412.7062},
      year={2015}
    }

    @article{papandreou15weak,
      title={Weakly- and Semi-Supervised Learning of a DCNN for Semantic Image Segmentation},
      author={George Papandreou and Liang-Chieh Chen and Kevin Murphy and Alan L Yuille},
      journal={arxiv:1502.02734},
      year={2015}
    }

Note that if you use the densecrf implementation, please consult and cite the following paper:

    @inproceedings{KrahenbuhlK11,
      title={Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials},
      author={Philipp Kr{\"{a}}henb{\"{u}}hl and Vladlen Koltun},
      booktitle={NIPS},      
      year={2011}
    }

### Performance

DeepLab currently achieves 73.9% on the challenging PASCAL VOC 2012 image segmentation task -- see the [leaderboard](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6). 

### Pre-trained models

We have released several trained models and corresponding prototxt files at [here](http://ccvl.stat.ucla.edu/software/deeplab/). Please check it for more model details.

The best model among the released ones yields 73.6% on PASCAL VOC 2012 test set.

### Experimental set-up

1. The scripts we used for our experiments can be downloaded from this [link](https://ucla.box.com/s/4grlj8yoodv95936uybukjh5m0tdzvrf):
    1. run_pascal.sh: the script for training/testing on the PASCAL VOC 2012 dataset. __Note__ You also need to download sub.sed script.
    2. run_densecrf.sh and run_densecrf_grid_search.sh: the scripts we used for post-processing the DCNN computed results by DenseCRF.
2. The image list files used in our experiments can be downloaded from this [link](https://ucla.box.com/s/rd9z2xvwsfpksi7mi08i2xqrj7ab4keb):
    * The zip file stores the list files for the PASCAL VOC 2012 dataset.
3. To use the mat_read_layer and mat_write_layer, please download and install [matio](http://sourceforge.net/projects/matio/files/matio/1.5.2/).

### FAQ

Check [FAQ](http://ccvl.stat.ucla.edu/deeplab_faq/) if you have some problems while using the code.

### How to run DeepLab

There are several variants of DeepLab. To begin with, we suggest DeepLab-LargeFOV, which has good performance and faster training time.

Suppose the codes are located at deeplab/code

1. mkdir deeplab/exper (Create a folder for experiments)
2. mkdir deeplab/exper/voc12 (Create a folder for your specific experiment. Let's take PASCAL VOC 2012 for example.)
3. Create folders for config files and so on.
    1. mkdir deeplab/exper/voc12/config  (where network config files are saved.)
    2. mkdir deeplab/exper/voc12/features  (where the computed features will be saved (when train on train))
    3. mkdir deeplab/exper/voc12/features2 (where the computed features will be saved (when train on trainval))
    4. mkdir deeplab/exper/voc12/list (where you save the train, val, and test file lists)
    5. mkdir deeplab/exper/voc12/log (where the training/test logs will be saved)
    6. mkdir deeplab/exper/voc12/model (where the trained models will be saved)
    7. mkdir deeplab/exper/voc12/res (where the evaluation results will be saved)
4. mkdir deeplab/exper/voc12/config/deeplab_largeFOV (test your own network. Create a folder under config. For example, deeplab_largeFOV is the network you want to experiment with. Add your train.prototxt and test.prototxt in that folder (you can check some provided examples for reference).)
5. Set up your init.caffemodel at deeplab/exper/voc12/model/deeplab_largeFOV. You may want to soft link init.caffemodel to the modified VGG-16 net. For example, run "ln -s vgg16.caffemodel init.caffemodel" at voc12/model/deeplab_largeFOV.
6. Modify the provided script, run_pascal.sh, for experiments. You should change the paths according to your setting. For example, you should specify where the caffe is by changing CAFFE_DIR. Note You may need to modify sub.sed, if you want to replace some variables with your desired values in train.prototxt or test.prototxt.
7. The computed features are saved at folders features or features2, and you can run provided MATLAB scripts to evaluate the results (e.g., check the script at code/matlab/my_script/EvalSegResults).

### Python

Seyed Ali Mousavi has implemented a python version of run_pascal.sh (Thanks, Ali!). If you are more familiar with Python, you may want to take a look at [this](https://github.com/TheLegendAli/CCVL). 
