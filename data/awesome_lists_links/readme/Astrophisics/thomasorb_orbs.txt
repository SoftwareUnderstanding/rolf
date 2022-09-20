# ORBS

[ORBS](https://github.com/thomasorb/orbs) (*Outil de Réduction Binoculaire pour* [SITELLE](http://www.cfht.hawaii.edu/Instruments/Sitelle)) is a data
reduction software created to process data obtained with
SITELLE. Is it the reduction software used by the CFHT.

Documentation may be found on https://readthedocs.org/projects/orbs/

## Installation


### Install ORB
   
[ORBS](https://github.com/thomasorb/orbs) depends on
[ORB](https://github.com/thomasorb/orb) which must be installed
first.

The archive and the installation instructions for
[ORB](https://github.com/thomasorb/orb) can be found on github

https://github.com/thomasorb/orb


### Install ORBS


#### Install specific dependencies

If you have followed the installation steps for orb, you already have a conda environment named `orb3`.
```bash
conda install -n orb3 -c conda-forge clint html2text distro lxml python-magic
conda activate orb3
pip install gitdb --no-deps 
pip install smmap --no-deps
pip install gitpython --no-deps
pip install cadcdata --no-deps
pip install cadcutils --no-deps
```

You will also need cfitsio. On Ubuntu you can install it with
``` bash
sudo apt install libcfitsio5 libcfitsio-bin
```
#### Install orbs module

During ORB install you have already created a folder name `orb-stable`. You can thus do

```bash
cd path/to/orb-stable
git clone https://github.com/thomasorb/orbs.git
python setup.py install # not for developer
```
**(developer only)**
```bash
cd
echo '/absolute/path/to/orb-stable/orbs' >> miniconda3/envs/orb3/lib/python3.7/site-packages/conda.pth
```

Test it:
```bash
conda activate orb3 # you don't need to do it if you are already in the orb3 environment
python -c 'import orbs.core'
```
