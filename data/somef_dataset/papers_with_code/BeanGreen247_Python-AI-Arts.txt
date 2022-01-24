# Python-AI-Arts
AI in Python that converts seleted image into the same image in an art style of another selected image.

## Setup
### Install python
```
sudo apt install python python-dev python3.7 python3.7-dev python-tk protobuf-compiler 
sudo apt install python-pip python3-pip 
sudo apt-get install python3 python-dev python3-dev \
     build-essential libssl-dev libffi-dev \
     libxml2-dev libxslt1-dev zlib1g-dev \
     python-pip
sudo apt-get install -y python-h5py
sudo apt-get install -y python3-h5py
```
### Upgrade setuptools
```
pip3 install --upgrade setuptools
```
Install testresources if missing
```
pip3 install testresources
```
### Install the dependencies
```
sudo apt-get install -y libpng-dev libtiff-dev libwebp-dev xcftools
pip3 install sklearn
pip3 install numpy
pip3 install argparse
pip3 install h5py==2.7.1
pip3 install keras==2.0.5
pip3 install conda
pip3 install pillow
pip3 install theano
pip3 install imread
pip3 install scipy==1.1.0
pip3 install scikit-image==0.15.0
pip3 install tensorflow==1.5
```
If a different version of scipy gets installed remove it and replace with 1.1.0
```
pip3 uninstall scipy
pip3 install scipy==1.1.0
```
Do the same for keras
```
pip3 uninstall keras
pip3 install keras==2.0.5
```
Do the same for h5py
```
pip3 uninstall h5py
pip3 install h5py==2.7.1
```
Do the same for scikit-image
```
pip3 uninstall scikit-image
pip3 install scikit-image==0.15.0
```
## Usage

There are 3 images to identify when we run the script

1. Your base image (to artify)
2. Your reference image (the art to learn from)
3. Your generated image

Run the following comand to generate an image in your chosen style
```
python3.6 Network.py "/path/to/content image" "path/to/style image" "result prefix or /path/to/result prefix"
```
Example
```
python3.6 Network.py ~/Pictures/AIArtConversion/scotland-castle.jpg ~/Pictures/AIArtConversion/modernism.jpg ~/Pictures/AIArtConversion/result.jpg
```
## Sample output
[MEGA.nz](https://mega.nz/#F!f9tTQKrI!QWYYvLEzRpvd8Mjn5Jt9iw)

## Convolutional Network Used
[VGG16-info](https://www.mathworks.com/help/deeplearning/ref/vgg16.html)

[VGG19-info](https://www.mathworks.com/help/deeplearning/ref/vgg19.html)

[Very Deep Convoltional Networks for Large-Scale Image Recognition by Karen Simonyan & Andrew Zisserman](https://arxiv.org/pdf/1409.1556.pdf)

Karen Simonyan & Andrew Zisserman. Very Deep Convoltional Networks for Large-Scale Image Recognition. PDF file. April 10 2015

[Generalizing Pooling Functions in Convolutional Neural Networks : Mixed, Gated, and Tree](https://arxiv.org/pdf/1509.08985.pdf)
