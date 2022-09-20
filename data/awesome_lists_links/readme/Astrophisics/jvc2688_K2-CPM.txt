# K2-CPM
Causal Pixel Model for K2 data

# How to use
```
$ python run_cpm.py

positional arguments:
  epic          int k2 epic number
  campaign      int campaign number, 91 for phase a, 92 for phase b
  n_predictors  int number of the predictors pixels
  l2            float strength of l2 regularization
  n_pca         int number of the PCA components to use, if 0, no PCA 
  distance      int distance between target pixel and predictor pixels
  exclusion     int how many rows and columns that are excluded around the target pixel
  input_dir     str directory to the target pixel files
  output_dir    str path to the output file

optional arguments:
  -p [pixel_list], --pixel [pixel_list]
                str path to the pixel list file that specify which pixels to be modelled. If not provided, the whole target pixel file will be modeled
```

# Example
```
$ python run_cpm.py 200069974 92 800 1e3 0 16 5 ./tpf ./output/200069974 -p ./test_pixel.dat
```

# The C++ alternative
*Suggestions about the C++ version are very welcome! Please contact Clément Ranc.*

## Introduction

The C++ version replaces the calculations performed when running
`cpm_part2.py`,  but not the ones done by `cpm_part1.py`. The C++
files must be compiled by hand prior using it for
the first time, as explained below.

## Files

The C++ version consists in the following files.

* `table.cpp` and the corresponding header file `table.h`: these files
define a  new class of one, two or three dimensional tables.

* `matrix.cpp` and the corresponding header file `matrix.h`: these
files define a new class  of square matrix that includes some very
commun operations such as Cholesky's  decomposition and a linear
system solver.

* `libcpm.cpp` and the corresponding header file `libcpm.h`: these
files corresponds to the  main code that should be executed.

* `Makefile`: this file allow compilation of the C++ version on most OS.

## Compilation

In linux and Mac OS, the C++ version can be compiled by the following commands:
```
$ cd path-to-main-directory/source/K2CPM/code/
$ make
```
where `path-to-main-directory` is the full path to the directory `K2-CPM` on your machine.

This `Makefile` will test the OS and adapt the compiler accordingly.
If the command `make` returns an error though, it might be because
your C++ compiler is not found or because you have not a C++ compiler
installed.

If the first case, please edit the lines 24-25 in `Makefile`:
```
# CC = your-own-compiler
# CFLAGS = -Wall -YourFlags
```
and uncomment them by removing `# `.

In the second case, it is necessary to install a C++ compiler. A free option is to 
install GCC 5 or later (see <https://gcc.gnu.org>).

## Usage

### Part 1

The first step is to follow the instructions provided in `tutorial.md`
up to the title CPM_PART2. Then, it is necessary to prepare the files
read by the C++ code. From a shell,
```
$ path-to-main-directory/source/K2CPM/code/
$ python conversion2cpp.py -p path-inputoutput
```
where `path-inputoutput/` is the path you are running the 
`tutorial.md`.

### Part 2

#### Without microlensing model

Then, the second part of the CPM will be run in C++. Once compiled,
the C++ version can be run directly, e.g. as follow,
```
$ path-to-main-directory/source/K2CPM/code/
$ ./libcpm path-inputoutput/ reference l2
```
where `path-to-main-directory` is the full path to the directory
`K2-CPM`, `path-inputoutput/` is the path you are using to follow
CPM_PART1 (see `tutorial.md`), `reference` is the content of the
variable `stem` from CPM_PART1 (see `tutorial.md`), including `_` at the end (if 
stem is `91_49_1022_119`, the reference is `91_49_1022_119_`) and `l2` is the
regularization strengh (e.g. 1000).

The above command run calculations and write the results is two files,
in the directory path-inputoutput/:

* `reference_results.dat`: solution of the linear system;

* `reference_cpmflux.dat`: the first column is the date, the second is
the target flux prediction,  the third is the difference flux.

#### Include a microlensing model (still beta version)

The current version may be used from a microlensing modeling
code. Only the *Part 2* may be affected by a microlensing model. 

For now, several files are used to interact with a modeling code. For
every position in the parameter space, a new file must be created in
the same directory we have run the *Part 1* (`path-inputoutput/`). If we come back to the
above example, the magnification at each epoch of the file
`91_49_1022_119_epochs_ml.dat` should be computed and saved in a new
file called `91_49_1022_119_magnification_ml.dat`. This file should
have the same number of lines than `91_49_1022_119_epochs_ml.dat` and
only one column, corresponding to the value of the magnification at
the  given epoch.

Also, it is necessary to say that we want to use a microlensing model to the
CPM Part 2. For that, the C++ code should be run as follow:
```
$ ./libcpm path-inputoutput/ reference l2 1
```
where the last integer `1` is a flag that makes the code to load the
magnification at each epoch.
