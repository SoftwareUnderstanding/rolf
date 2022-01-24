###  Clone of pytorch-encoding created by [Hang Zhang](http://hangzh.com/):
The [Synchronized Batchnorm](http://hangzh.com/PyTorch-Encoding/) package needs installation of Pytorch from source. Since it is not maintained by the PyTorch team, it is not always up-to-date with PyTorch master branch.
In order to avoid compatibility issues, this is a modified copy of the repo, and works by following the instructions below. If you use this for your work, please cite the original paper.

1. Download and install [Anaconda Python 3.6](https://www.anaconda.com/download/). Alternatives are ofcourse possible.
2. Create an environment and install dependencies.
    ```
    conda create -n myenv python=3.6
    source activate myenv
    
    # Needed to install Pytorch
    conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing requests future
    conda install -c mingfeima mkldnn
    conda install -c pytorch magma-cuda90
    ```
3. Install PyTorch from source, but revert to a specific commit first to avoid incompatibilities.
    ```
    # Clone the repo
    git clone --recursive https://github.com/pytorch/pytorch.git
    cd pytorch
    
    # Revert to specific commit
    git reset --hard 13de6e8
    git submodule update --init
    
    # Only for Volta GPUs
    export TORCH_CUDA_ARCH_LIST="3.5;5.0+PTX;6.0;6.1;7.0"

    # Install pytorch - be patient please.
    python setup.py install
    cd ..

    ```
4. Install torchvision and ninja.
    ```
    # Install torchvision from source
    git clone https://github.com/pytorch/vision.git
    cd vision
    python setup.py install
    cd ..
    
    # Get ninja
    wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
    unzip ninja-linux.zip
    mv ninja /path/to/anaconda/bin
    ```

5. Install Synchronized Batchnorm package (my local copy, with the hacks needed to make it work). 
    ```
    # Install pytorch encoding
    git clone https://github.com/kmaninis/pytorch-encoding.git
    cd pytorch-encoding
    python setup.py install
    cd ..
    ```


# PyTorch-Encoding - Original README.md

created by [Hang Zhang](http://hangzh.com/)

## [Documentation](http://hangzh.com/PyTorch-Encoding/)

- Please visit the [**Docs**](http://hangzh.com/PyTorch-Encoding/) for detail instructions of installation and usage. 

- Please visit the [link](http://hangzh.com/PyTorch-Encoding/experiments/segmentation.html) to examples of semantic segmentation.

## Citations

**Context Encoding for Semantic Segmentation** [[arXiv]](https://arxiv.org/pdf/1803.08904.pdf)  
 [Hang Zhang](http://hangzh.com/), [Kristin Dana](http://eceweb1.rutgers.edu/vision/dana.html), [Jianping Shi](http://shijianping.me/), [Zhongyue Zhang](http://zhongyuezhang.com/), [Xiaogang Wang](http://www.ee.cuhk.edu.hk/~xgwang/), [Ambrish Tyagi](https://scholar.google.com/citations?user=GaSWCoUAAAAJ&hl=en), [Amit Agrawal](http://www.amitkagrawal.com/)
```
@InProceedings{Zhang_2018_CVPR,
author = {Zhang, Hang and Dana, Kristin and Shi, Jianping and Zhang, Zhongyue and Wang, Xiaogang and Tyagi, Ambrish and Agrawal, Amit},
title = {Context Encoding for Semantic Segmentation},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```

**Deep TEN: Texture Encoding Network** [[arXiv]](https://arxiv.org/pdf/1612.02844.pdf)  
  [Hang Zhang](http://hangzh.com/), [Jia Xue](http://jiaxueweb.com/), [Kristin Dana](http://eceweb1.rutgers.edu/vision/dana.html)
```
@InProceedings{Zhang_2017_CVPR,
author = {Zhang, Hang and Xue, Jia and Dana, Kristin},
title = {Deep TEN: Texture Encoding Network},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {July},
year = {2017}
}
```
