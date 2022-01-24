###
# ICLUST
###

ICLUST is a software for unsupervised clustering of images based on transfer
learning using a pre-trained keras NN model.

# Installation

check that $PYTHONPATH=/Users/KevinBu/tools/sandbox/lib/python3.7/site-packages/
python3 setup.py install

or on hpc
module load python/3.7.3 && export PYTHONPATH=$PYTHONPATH:/hpc/users/buk02/tools/sandbox/lib/python3.7/site-packages/
python setup.py install --prefix=/hpc/users/buk02/tools/sandbox

Further information can be found at the original github. The code was adapted
and forked from the original repository in late 2018.
https://github.com/elcorto/imagecluster.git

