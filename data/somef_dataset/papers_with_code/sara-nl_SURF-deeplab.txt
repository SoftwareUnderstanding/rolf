# Extended FoV Semantic Segmentation
This repository is developed as part of the Examode EU project, and is meant for conducting experiments for large field-of-view semantic segmentation. The current codebase supports CAMELYON16 and CAMELYON17, and supports efficient execution on multi-node CPU clusters, as well as multi-node, multi-GPU clusters. Models using very large FoV (> 1024x1024) can be trained on multi-GPU cluster, using the instructions below. The models adapted for the use case of semantic segmentation of malignant tumor regions are:
- EfficientDet ( https://arxiv.org/abs/1911.09070 )
- DeeplabV3+ ( https://arxiv.org/abs/1802.02611 )

https://camelyon16.grand-challenge.org/

https://camelyon17.grand-challenge.org/

# Setup
These steps ran on LISA with this module environment: 

Modules loaded:
```
cd $HOME
module purge
module load 2019
module load 2020
module load Python/3.8.2-GCCcore-9.3.0
module load OpenMPI/4.0.3-GCC-9.3.0
module load cuDNN/7.6.5.32-CUDA-10.1.243
module load NCCL/2.5.6-CUDA-10.1.243
module unload GCCcore
module unload ncurses
module load CMake/3.11.4-GCCcore-8.3.0

```

## Dependencies
We will create a virtual environment with Openslide (https://openslide.org/api/python/) and libvips (https://libvips.github.io/libvips/install.html), for opening and sampling from whole-slide-images.

- Pick a name for the virtual environment, and make the virtual environment folder using `virtualenv`:

```
VENV_NAME=openslide
cd $HOME
virtualenv $HOME/virtualenvs/$VENV_NAME
cd $HOME/virtualenvs/$VENV_NAME
```
Then add the relevant values to the environment variables:
```
export PATH=$HOME/virtualenvs/$VENV_NAME/bin:$PATH
export LD_LIBRARY_PATH=$HOME/virtualenvs/$VENV_NAME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/virtualenvs/$VENV_NAME/lib:$LD_LIBRARY_PATH
export CPATH=$HOME/virtualenvs/$VENV_NAME/include:$CPATH
```

### LibTIFF
1. Download a release from the official repository and untar
```
wget http://download.osgeo.org/libtiff/tiff-4.0.10.tar.gz
tar -xvf tiff-4.0.10.tar.gz
```
2. Build and configure the LibTIFF code from the inflated folder
```
cd $HOME/virtualenvs/$VENV_NAME/tiff-4.0.10
CC=gcc CXX=g++ ./configure --prefix=$HOME/virtualenvs/$VENV_NAME
make -j 8
```
3. Install LibTIFF
```
make install
cd ..
```

### OpenJPEG
The official install instructions are available [here](https://github.com/uclouvain/openjpeg/blob/master/INSTALL.md).
1. Download and untar a release from the official repository
```
wget https://github.com/uclouvain/openjpeg/archive/v2.3.1.tar.gz
tar -xvf v2.3.1.tar.gz
```
2. Build the OpenJPEG repository code
```
cd $HOME/virtualenvs/$VENV_NAME/openjpeg-2.3.1
CC=gcc CXX=g++ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/virtualenvs/$VENV_NAME -DBUILD_THIRDPARTY:bool=on
make -j 8

```
3. Install OpenJPEG (we already added the paths to the environment variables)
```
make install
cd ..
```

### OpenSlide
1. Download and untar a release from the official repository
```
wget https://github.com/openslide/openslide/releases/download/v3.4.1/openslide-3.4.1.tar.gz
tar -xvf openslide-3.4.1.tar.gz
```
2. Build and configure the OpenSlide code
```
cd $HOME/virtualenvs/$VENV_NAME/openslide-3.4.1
CC=gcc CXX=g++ PKG_CONFIG_PATH=$HOME/virtualenvs/$VENV_NAME/lib/pkgconfig ./configure --prefix=$HOME/virtualenvs/$VENV_NAME
make -j 8
```
3. Install OpenSlide (we already added the paths to the environment variables)
```
make install
cd ..
```

### LibVips
1. Download and untar a release from the official repository
```
wget https://github.com/libvips/libvips/releases/download/v8.9.2/vips-8.9.2.tar.gz
tar -xvf vips-8.9.2.tar.gz
```
2. Build and configure the Libvips code
```
cd $HOME/virtualenvs/$VENV_NAME/vips-8.9.2
CC=gcc CXX=g++ PKG_CONFIG_PATH=$HOME/virtualenvs/$VENV_NAME/lib/pkgconfig ./configure --prefix=$HOME/virtualenvs/$VENV_NAME
make -j 8
```
This step may lead to errors. Use 
```
module load pre2019, module load cmake/2.8.11
```
to solve
3. Install Libvips (we already added the paths to the environment variables)
```
make install
cd ..
```

### Setting up the Python depencies (specific to LISA GPU)
Now export environment variables for installing Horovod w/ MPI for multiworker training, and install Python packages:
```
module purge
module load 2019
module load 2020
module load Python/3.8.2-GCCcore-9.3.0
module load OpenMPI/4.0.3-GCC-9.3.0
module load cuDNN/7.6.5.32-CUDA-10.1.243
module load NCCL/2.5.6-CUDA-10.1.243
module unload GCCcore
module unload ncurses
module load CMake/3.11.4-GCCcore-8.3.0
source $HOME/virtualenvs/openslide-py38/bin/activate

export HOROVOD_CUDA_HOME=$CUDA_HOME
export HOROVOD_CUDA_INCLUDE=$CUDA_HOME/include
export HOROVOD_CUDA_LIB=$CUDA_HOME/lib64
export HOROVOD_NCCL_HOME=$EBROOTNCCL
export HOROVOD_GPU_ALLREDUCE=NCCL
export HOROVOD_GPU_BROADCAST=NCCL
export HOROVOD_WITHOUT_GLOO=1
export HOROVOD_WITH_TENSORFLOW=1
export PATH=/home/$USER/virtualenvs/openslide-py38/bin:$PATH
export LD_LIBRARY_PATH=/home/$USER/virtualenvs/openslide-py38/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/$USER/virtualenvs/openslide-py38/lib:$LD_LIBRARY_PATH
export CPATH=/home/$USER/virtualenvs/openslide-py38/include:$CPATH
# Export MPICC
export MPICC=mpicc
export MPICXX=mpicpc
export HOROVOD_MPICXX_SHOW="mpicxx --showme:link"

# Install Python packages
pip install -r requirements.txt
```

- Options for model training:
For model training two architectures are used:
1. `DeeplabV3+`   ( ~ 41 mln parameters, 177B FLOPs)
2. `EfficientDetD[0-7]` ( ~ 20 mln parameters for D4, 18B FLOPs)

Please look in repositories for further steps. 

## Research
If this repository has helped you in your research we would value to be acknowledged in your publication.

# Acknowledgement
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 825292. This project is better known as the ExaMode project. The objectives of the ExaMode project are:
1. Weakly-supervised knowledge discovery for exascale medical data.  
2. Develop extreme scale analytic tools for heterogeneous exascale multimodal and multimedia data.  
3. Healthcare & industry decision-making adoption of extreme-scale analysis and prediction tools.

For more information on the ExaMode project, please visit www.examode.eu. 

![enter image description here](https://www.examode.eu/wp-content/uploads/2018/11/horizon.jpg)  ![enter image description here](https://www.examode.eu/wp-content/uploads/2018/11/flag_yellow.png) <img src="https://www.examode.eu/wp-content/uploads/2018/11/cropped-ExaModeLogo_blacklines_TranspBackGround1.png" width="80">

