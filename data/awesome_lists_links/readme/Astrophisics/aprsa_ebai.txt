# EBAI: ECLIPSING BINARIES via ARTIFICIAL INTELLIGENCE

November 1, 2008

## 1. INTRODUCTION

EBAI -- Eclipsing Binaries via Artificial Intelligence -- is a project aimed at automating the process of solving light curves of eclipsing binary stars. EBAI is based on the back-propagating neural network paradigm and is highly flexible in construction of neural networks. The design and concepts are explained in detail by Prsa et al. (2008), ApJ 687, 542.

## 2. COMPILING

EBAI comes in two flavors: serial (ebai) and multi-processor (ebai.mpi). To compile either, you must be running a GNU-compatible build system. This means that ebai and ebai.mpi will build readily on any linux platform; the code should be readily portable to other platforms, but this has not been tested yet. Any success reports would be greatly appreciated!

To compile the multi-processor version of EBAI, you need to have all MPI libraries and build system installed, mpicc in particular.

Depending on the version of EBAI you want to use, issue `make` in either src/ subdirectory (serial version) or mpi/ subdirectory (parallel version).

To install EBAI, copy the executable (ebai or/and ebai.mpi) to /usr/local/bin or any alternative directory in the bin path.

## 3. USAGE

EBAI can be run in training, continued training and recognition mode. This is determined by the appropriate switch: -t for training, -c for continued training, and -r for recognition.

In this tarball, 10 light curves are provided as an example. These light curves are polyfit data sampled in 201 equidistant phase steps. Every input to EBAI is a single column with the number of data points on the input layer, followed by an empty line, followed by the number of parameters mapped.

Since it is easiest to learn from an example, here is how EBAI would be trained on 10 light curves in the example/ subdirectory, each light curve having 201 points mapped onto 5 parameters (see the ApJ paper for details):

./ebai -t -i 5000 -s 10 -n 201:20:5 --data-dir ../example \
       --i2h i2h.matrix --h2o h2o.matrix --param-bounds bounds.data \
       > training.dat

The '-t' switch invokes the training mode; '-i 5000' sets the number of iterations to 5000; '-s 10' sets the sample size to 10 light curves; '-n 201:20:5' sets the neural network topology to 201 input nodes, 20 hidden nodes and 5 output nodes; all other parameters determine I/O files for the results. The output is redirected to 'training.dat' where the learning curve is printed out.

Let us now run the code in recognition mode, on the same data-set:

./ebai -r -s 10 -n 201:20:5 --data-dir ../example --i2h i2h.matrix \
      --h2o h2o.matrix --param-bounds bounds.data > recognition.dat

The output file 'recognition.dat' will contain 10 columns, original and recognized value of each of the 5 parameters.

A manual for using EBAI will be made available shortly; in the mean time use the method of trial-and-error for fine-tuning EBAI.

## 4. CONTACT:

In case you have any questions, please feel free to contact me at:

	aprsa@villanova.edu
