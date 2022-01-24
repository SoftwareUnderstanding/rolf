Adaptation of NICE for inpainting missing data datasets
=======================================================

This repository extends the original NICE code to support inpainting datasets with general missing data
supplied in the form of mask images.



Training a model follows the instructions in the original paper- after installing all dependencies, call [`pylearn2/scripts/train.py`](https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/scripts/train.py) on  
`exp/nice_mnist.yaml`

The missing data dataset should have one `index.txt` file listing each corrupted image file with its label in Caffe ImageData format, and one `index_mask.txt` file in the same format listing the mask file corresponding to each
corrupted image file.

To inpaint such a dataset, run (for MNIST):
`` python pylearn2/scripts/mnist_inpainting.py exp/nice_mnist_best.pkl <missing_data_dir> ``

Where the `.pkl` file is the model generated through training.

The new dataset will be created at the same location as the missing dataset, under the same name with the addition of `_nice_ip`.

NICE: Non-linear independent components estimation
==================================================

This repository contains code (in [`pylearn2/`](https://github.com/laurent-dinh/nice/blob/master/pylearn2/)) and hyperparameters (in [`exp/`](https://github.com/laurent-dinh/nice/blob/master/exp/)) for the paper:

["NICE: Non-linear independent components estimation"](http://arxiv.org/abs/1410.8516) Laurent Dinh, David Krueger, Yoshua Bengio. ArXiv 2014.

Please cite this paper if you use the code in this repository as part of
a published research project.

We are an academic lab, not a software company, and have no personnel
devoted to documenting and maintaing this research code.
Therefore this code is offered with minimal support.
Exact reproduction of the numbers in the paper depends on exact
reproduction of many factors,
including the version of all software dependencies and the choice of
underlying hardware (GPU model, etc). We used NVIDA Ge-Force GTX-580
graphics cards; other hardware will use different tree structures for
summation and incur different rounding error. If you do not reproduce our
setup exactly you should expect to need to re-tune your hyperparameters
slight for your new setup.

Moreover, we have not integrated any unit tests for this code into Theano
or Pylearn2 so subsequent changes to those libraries may break the code
in this repository. If you encounter problems with this code, you should
make sure that you are using the development branch of [Pylearn2](https://github.com/lisa-lab/pylearn2/) and
[Theano](https://github.com/Theano/Theano/),
and use `git checkout` to go to a commit from approximately October 21, 2014. More precisely [`git checkout 3be2a6`](https://github.com/lisa-lab/pylearn2/commit/3be2a6d5ff81273c12023208166b630300eff338) (for Pylearn2) and [`git checkout 165eb4`](https://github.com/Theano/Theano/commit/165eb4e66ab1f5320b2fe67c630a7e76ae5e6526) (for Theano).

This code itself requires no installation besides making sure that the
`nice` directory is in a directory in your PYTHONPATH. If
installed correctly, `python -c "import nice"` will work. You
must also install Pylearn2 and Pylearn2's dependencies (Theano, numpy,
etc.)

Call [`pylearn2/scripts/train.py`](https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/scripts/train.py)
on the various yaml files in this repository
to train the model for each dataset reported in the paper. The names of
*.yaml are fairly self-explanatory.