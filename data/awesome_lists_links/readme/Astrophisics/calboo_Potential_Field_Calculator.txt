# Potential Field Calculator

## Purpose

bootsik.f90 is a FORTRAN90 code that can be used to generate a potential magnetic field on a 3D mesh, staggered relative to the magnetic potential, by extrapolating the magnetic field normal to the photospheric surface. The code first calculates a magnetic potential using a modified Green’s function method and then uses a finite differencing scheme to calculate the magnetic field from the potential. The IDL script boobox.pro can then be used to visualise the magnetic field.

## Files

| File | Description |
| --- | --- |
| Boobox.pro   | IDL script for visualising the 3D potential field  |
| bootsik      | The compiled FORTRAN90 program                     |
| bootsik.f90  | The FORTRAN90 code                                 |
| bootsik.sh   | Submission script for parallel job using QMUL HPC  |
| bx/by/bz.dat | Output files specifying the potential field        |
| bz0.dat      | Input file with normal field at the photosphere    |
| constants.mod| FORTRAN module file for constants                  |

## Inputs and Usage

### Bootsik - Potential field Calclator

The input data for this code is the magnetic field normal to the lower photospheric boundary. This can be either specified within the code itself or read in from an input file. There are examples of two analytically defined inputs in the code as well as a block that can be used to import data. The user can uncomment whichever Bz0 profile they want to use or alternatively specify their own Bz0 field.

If an analytic input is used then the parameters can be set in the module *constants* at the top of the f90 file. If an input file is used the user should be careful to make sure that the size of the imported array is equal to the size of bz0 as defined in *constants*. There is an example input file *Bz0.dat* provided which has dimensions of nx=308, ny=308.

The user can also decide whether to use standard or extended input which is done by uncommenting the chosen line in *constants*. The standard input uses an input area equal in size to the base of the domain being calculated whereas the extended input option uses an input area three times larger in each direction. The example *Bz0.dat* data is ideal for use with the extended input when nx=ny=100.

### Boobox.pro - Magnetic field visualiser

The input data for this IDL script can either be output files from the potential solver *bootsik* or output from the MHD solver Lare3d (https://warwick.ac.uk/fac/sci/physics/research/cfsa/people/tda/larexd/). The user will need to uncomment the right parts of the script depending on which type of input is being used. In either case the domain dimensions nx, ny and nz must be correctly specified. 

The user can also decide on the viewing angle by setting ax and az, the tick labels on the axis by setting min and max values in each direction and how the field lines are drwn by setting the value of *grid*. 

| grid value | field lines setup |
| --- | --- |
| 1  | starting points distributed uniformly at the photosphere |
| Any other value | starting points randomly distributed throughout domain |

## Methods

For an in depth explanation of the methods used please see the publication, *Potential magnetic field calculator for solar physics applicationsusing staggered grids*:

here: https://arxiv.org/pdf/1903.10546.pdf   
or here: https://www.aanda.org/articles/aa/full_html/2019/05/aa34684-18/aa34684-18.html

## Compiling and Running

bootsik.f90 can be run on a single machine. In this case use the following lines to compile and then simply run the job:  
**gfortran -O3 bootsik.f90**

Alternatively you can run the code in parallel for improved runtimes.
To run the code in parallel on a HPC use the following lines to compile:  
**module load intel/2018.1  
ifort -o bootsik -O3 -parallel bootsik.f90 -xHost**

Then submit the job to a HPC cluster. Intel 2018 works well but other compilers that can auto parallelise may also work.

To use boobox, simply open an IDL window and run enter the command **boobox**, making sure that the dat files are in the current directory. If using Lare3d input the data snapshot will also need to be specified. This can be done by first loading the snapshot e.g. **ds = getdata(10)** and then running boobox with **boobox, ds**.
