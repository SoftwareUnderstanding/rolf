SL1M
====

###Radio Synthesis Deconvolution using L1 Minimisation

This is an implementation of the algorithm described in ["Direct deconvolution of radio synthesis images using L1 minimisation", Stephen J Hardy, A&A 557, A134 (2013)](http://www.aanda.org/articles/aa/abs/2013/09/aa21833-13/aa21833-13.html).

This paper was written over several months in the beginning of 2013 during an extended period of leave. It demonstrates a new method for deconvolving radio synthesis images (extending work by Li Corwell and De Hoog - see the paper for details!!).

Due to the compressed time period over which this work was done, this implementation is not integrated into any astronomical data reduction system. Ideally it would be callable from [CASA](http://casa.nrao.edu), but determining the structure of the CASA measurement set format, and the correct place to implement the algorithm in the casa pipeline, was taking me too long, so I opted for a more direct (but far less elegant and useable) approach.

Anyone who would like to assist me in interegrating this into a tool chain, even just advising where to insert the relevant code, would be most welcome to contact me.

###Installation

The code can currently only be run on a machine with 2 CUDA devices, such as an Amazon GPU instance.

The easiest way to exercise the code is to spin up a GPU instance on Amazon using the the machine image: ami-2ac85b43 

To install the software, clone the repository 

`git clone git://github.com/StephenJHardy/SL1M SL1M`

and download [tclap](http://sourceforge.net/projects/tclap/files/tclap-1.2.1.tar.gz/download) which is a simple command line parser written in C++. Run make and you should be ready.

###Input file format

The code takes a binary input format that should be output from whatever package that you are using after removing noisy records, calibration and continuum subtraction. 

Files are either float or double depending on the version of the code you are using (the switch is in config.h). I'll assume float here.

The input file consists of:

* int - number of records
* int - number of channels
* float - frequency of the 0th channel
* float - channel width
* nrec * float - u visibility coordinate of each record
* nrec * float - v visibility coordinate of each record
* nrec * float - w visibility coordinate of each record
* nrec * nchan * float - real part of the visibility of each record
* nrec * nchan * float - imaginary part of the visibility of each record

A sample input file from ngc5921 is included in the checkout. It is 63 channels with 11934 records derived from the tutorial casa file referenced in the above paper.

###Output file format

The output file is a simple binary image file format:

* int - width
* int - height
* width * height * float - image data

###Usage
```
USAGE: 

   ./sl1m  [-i <string>] [-o <string>] -l <float> [-s <float>] [-p <float>]
           [-c <int>] [-m <int>] [-r <int>] [-t <int>] [-k <string>] [-a
           <string>] [-f] [-g <float>] [--] [--version] [-h]


Where: 

   -i <string>,  --input <string>
     Input file name

   -o <string>,  --output <string>
     Output file name

   -l <float>,  --lambda <float>
     (required)  regularisation parameter for L1 minimisation

   -s <float>,  --size <float>
     Size of output image (assumed square)

   -p <float>,  --pixel <float>
     pixel size in arc seconds

   -c <int>,  --channel <int>
     Channel number to process

   -m <int>,  --maxiters <int>
     Maximum number of FISTA steps to take

   -r <int>,  --records <int>
     Max records to process (0 = all)

   -t <int>,  --threshold <int>
     discard visibilities greater than this size (0.0 = use all)

   -k <string>,  --initialout <string>
     output the initial approximation to this file

   -a <string>,  --initialin <string>
     read initial appoximation from this file

   -f,  --fftinit
     use FFT to calculate initial approximation

   -g <float>,  --gaussian <float>
     gaussian component size in pixels (0.0 = delta function pixels)

   --,  --ignore_rest
     Ignores the rest of the labeled arguments following this flag.

   --version
     Displays version information and exits.

   -h,  --help
     Displays usage information and exits.


   SL1M - synthesis through L1 minimisation
```
###Example

To process the NGC5921 data that ships with the archive, run 

```
./sl1m -l 1 -s 256 -p 60 -m 100 -f -i ngc5921.bin -c 32
```
This will produce a 256 by 256 image with 60 arc second pixels after 100 iterations of SL1M, initial conditions set by a gridded FFT, from channel 32 of the input data. Lambda is set to 1.

###License

The code is licensed under the BSD two clause license. No warranty ... use at own risk ... yada yada yada


###Future

An implementation of the code using OMP exists and was used for cross checking a validating the code here. A version using MPI for multiprocessing across multiple machines is partly implemented.