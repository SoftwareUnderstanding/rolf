dst: python parallel destriper
==============================

## requirements

* `h5py` - `python-h5py` in ubuntu
* `PyTrilinos` - `python-pytrilinos` in ubuntu
* `cython` - `cython` in ubuntu
* `openmpi` - `openmpi-bin` in ubuntu

## install

* download the package from github
* run:

        python setup.py build_ext -i #builds cython extensions in place

## run

* download the 3 days test dataset from figshare `ch6_fill_3d.h5` 
  and copy inside the `h5/` folder
* try running the destriping serially:

        python dst.py ch6_256.cfg

* try running in parallel:

        mpirun -np 3 python dst.py ch6_256.cfg

## input data format

currently data needs to be in a single HDF5 file with columns:

     dtype=[('TIME', '>f8'), ('PHI', '>f8'), ('THETA', '>f8'), ('PSI', '>f8'), ('FLAG', '>i2'), ('TEMP', '>f4'), ('Q', '>f4'), ('U', '>f4')])

in order to support other data format it is possible to modify `dst_io.read_data`

## configuration file format

see comments in the example configuration file `ch6_256.cfg`

## related scientific publication

Scientific paper: "Destriping CMB polarimeter data", http://arxiv.org/abs/1309.5609

Destriped maps from the B-Machine 40GHz polarimeter are available on
figshare at: http://dx.doi.org/10.6084/m9.figshare.644507

## source code 

* source code is released under GPLv3
* source repository: http://github.com/zonca/dst
* author: Andrea Zonca, http://about.me/andreazonca

If you use this software or part of it for a scientific publication,
you are required to share your code under GPLv3 and cite the above paper.
