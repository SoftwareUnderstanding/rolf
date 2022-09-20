## calclens

Curved-sky grAvitational Lensing for Cosmological Light conE simulatioNS  
Copyright (C) 2009-2016 Matthew R. Becker  
Released under GNU GPL v3 - see COPYING and AUTHORS for details.  

CALCLENS stands for Curved-sky grAvitational Lensing for Cosmological
Light conE simulatioNS. It is a curved-sky multiple-plane ray tracing
code for generating weak gravitational lensing shear fields from light
cone simulations. Currently only flat LCDM models are supported.

License and Conditions of Use
-----------------------------
This code is released publicly under GNU GPL v3 (see COPYING for
details). The standard GNU disclaimer (which most definitely applies
here!) is

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

If CALCLENS is used for any scientific work, please
cite the paper describing the code 

    Becker 2013, MNRAS, 435, 115 [arXiv:astro-ph/1210.3069]

and also post your own work (and any updates) to the arXiv
(http://arxiv.org/).

CALCLENS is a BYOLC code, which stands for Bring Your Own Light
Cone. Utilities for reading light cone formats (see `lightconeio.c` for
how to write your own) are included below.

Compilation
-----------
CALCLENS requires the following widely available public software
libraries:

GSL - GNU Scientific Library (http://www.gnu.org/software/gsl/)  
CFITSIO - for read and writing FITS files (http://heasarc.gsfc.nasa.gov/fitsio/)  
HDF5 v1.8 - for the indexed light cone format (http://www.hdfgroup.org/HDF5/)  
FFTW3 - for FFTs used by SHTs (http://www.fftw.org/)  

Other requirements include:
    
1. a 64 bit system - the types long and double in C must both be 8 bytes
2. a Unix/Linux-like environment (It has not been tested on Mac OSX.)

Once these libraries are installed (or if you have them already) and
requirements are met, then you just have to edit the Makefile to match
your configuration.  For Gadget-2 users, this should be very
intuitive.  Here is an example for the KIPAC Orange cluster

    ifeq ($(COMP),"orange")
    CC          =  mpicc
    OPTIMIZE    =  -g -O3
    GSLI        =  -I/afs/slac.stanford.edu/g/ki/software/gsl/1.15/include
    GSLL        =  -L/afs/slac.stanford.edu/g/ki/software/gsl/1.15/lib
    FFTWI       =  -I/afs/slac.stanford.edu/g/ki/software/fftw/3.3/include 
    FFTWL       =  -L/afs/slac.stanford.edu/g/ki/software/fftw/3.3/lib
    HDF5I       =  -I/afs/slac.stanford.edu/g/ki/software/hdf5/1.8.8/include 
    HDF5L       =  -L/afs/slac.stanford.edu/g/ki/software/hdf5/1.8.8/lib
    FITSI       =  -I/afs/slac.stanford.edu/g/ki/software/cfitsio/3.29/include
    FITSL       =  -L/afs/slac.stanford.edu/g/ki/software/cfitsio/3.29/lib
    EXTRACFLAGS =
    EXTRACLIB   =
    endif

Then simply type "make" in the source directory and you should get an
executable named "raytrace" for your enjoyment.  

Running an MPI code
-------------------
CALCLENS is an MPI-1 parallel code. See the local user guide (or ask a
friend) on how to run MPI code on your local compute cluster.
Typically one needs to make a job script and the include commands like 

    mpirun <path to calclens> <path to config file>

to get this work. See the Ray Tracing below for how to actually run
the code.

Ray Tracing 
-----------
Ray tracing a light cone involves two steps.  First, the light cone
must be rewritten into the HDF5 index light cone format.  Then it can
be ray traced.  

1) Making the indexed light cone files.
   
   Code to make lens planes is in the lensplanes directory. You will 
   need to setup the configuration file properly as described in Configuration
   below.
   
   Making lens planes is a purely serial operation and requires a lot
   of memory (though you do not need to have the *entire* light cone in
   memory). The best way to make lens planes is to run an MPI job with
   as many tasks as you have cores per node.  Task 0 will make the
   lens planes and then at the end all of the tasks read the data to
   make maps of the matter density for error checking. This can take a
   while depending on how much data you have, but only has to be done once.

2) Ray Tracing
   
   The ray tracing code is included in the main diretcory. You will need 
   to setup your configuration file properly (see Configuration) and 
   then run the code as described above.

   I would recommend, depending on resolution, at least 128 cores for
   220 deg^2 patch and then small increases after that as the area gets
   larger. Note that the SHT steps are the same cost in CPU hours
   regardless of area, but that for large enough areas the MG steps
   dominate the running time.

Configuration
------------------
CALCLENS uses a configuration file to set most options for running the
code.  Options *not* set in the configuration file, but in the
Makefile, include

    BORNAPPRX - ray trace with born approximation
    OUTPUTRAYDEFLECTIONS - output ray deflections in ray output
    OUTPUTPHI - output lensing potential in ray output
    USE_FITS_RAYOUT - set to use fits for writing rays, otherwise will
                      use a pure binary format
    USE_FULLSKY_PARTDIST - set to tell the code to use a full-sky
                      particle distribution in the SHT step, but only
                      use the specified area for the MG step
    SHTONLY - set to force the code to use SHTs only

If any of these options are changed, the code must be recompiled.

See raytrace.cfg for an example typical configuration file. Its basic
structure is a set of tag-value pairs like this 

    OmegaM 0.27

The tag names are not case sensitive. All units are assumed to be in
comoving Mpc/h, Msun/h, MB, degrees or seconds. Comments are denoted by '#',
can appear anywhere on a line except between a tag and its value
(i.e. "OmegaM #this is a comment 0.25" is *not* allowed), and are
strongly encouraged.

The options are as follows.

    WallTimeLimit - time limit for code in seconds
    WallTimeBetweenRestart - time between writing of restart files

    OmegaM - matter density in units of critical at z = 0
    maxComvDistance - maximum comoving distance to end of light cone
    NumLensPlanes - number of lens planes

Note that maxComvDistance and NumLensPlanes should be set so that each
lens plane is approximately 25 - 30 Mpc/h.

     LensPlanePath - path to lens planes
     LensPlaneName - base name of lens plane

When writing lens planes, this path and base name are used to
construct the lens plane files like this

    <LensPlanePath>/<LensPlaneName>XXXX.h5

where XXXX is the lens plane number (i.e. 0010 for plane 10).

    OutputPath - path to directory for code output
    RayOutputName - base name of ray outputs

Comment out RayOutputName (by adding a '#' in front of it) to prevent rays
from being written to disk.  These outputs can be many TB for most
calculations so this is usually not recommended.

    NumRayOutputFiles - # of files to output rays into
    NumFilesIOInParallel - # of files which do I/O at the same time

NumFilesIOInParallel must be less than both NumRayOutputFiles and
NumGalOutputFiles. The ray outputs are written to disk like this

    <OutputPath>/<RayOutputName>XXXX.YYYY

where XXXX is the lens plane number (from 0 to NumLensPlanes-1) and
YYYY is a file index from 0 to NumFilesIOInParallel-1.  Note that the
code will write an additional ray data output with lens plane number
NumLensPlanes which has the rays at the very edge of the light cone
(i.e. at maxComvDistance).  The rest of the ray outputs are at
comoving distance 

    x*dC + dC/2.0 

where x is the lens plane number and dC is the lens plane width given
by (maxComvDistance/NumLensPlanes).  

The rays, ray tracing area, and Poisson solver are controlled by 

    bundleOrder - HEALPix order for bundle cells (usually 6 or 7)
    rayOrder - HELAPix order for rays (somewhere between 14-16)
    minRa - minimum ra for ray tracing area in degrees
    maxRa - maximum ray for ray tracing area in degrees
    minDec - min dec for ray tracing area in degrees
    maxDec - max dec for ray tracing area in degrees
    HEALPixRingWeightPath - path to data directory from public HELAPix
    SHTOrder - HEALPix order of SHT
    ComvSmoothingScale - a comoving smoothing scale (usually ~ a few
                         N-body softening lengths)

Optionally one can set the parameter

    MGConvFact - ratio of residual norm to truncation error on finest
                 MG level 

to control the convergence of the MG code. The code has built in
defaults (0.1) so changing this parameter is *not* recommended.

CALCLENS uses the quadrature weights from the public HEALPix package
is HEALPixRingWeightPath is specified. Note that if you make minRa
greater than maxRa, then CALCLENS will wrap the domain around the
sphere in the other direction (i.e. minRa = 360 and maxRa = 5 will set
the domain to go from -5 to +5 degrees).

CALCLENS optionally does a grid search for the lensed images of galaxies placed
in the light cone. The source galaxies must be in FITS binary tables
in extension 1 with tags px, py and pz which have the comoving
location of the galaxy in the light cone in Mpc/h.  An ID tag is
recommended but is not currently used by the code. To tell CALCLENS
about the galaxies, set the following options

    GalsFileList - file containing list of source galaxy files
    GalOutputName - base output name for galaxy images
    NumGalOutputFiles - number of output files for galaxy images

The galaxy output files are formatted like the ray output files with

    <OutputPath>/<GalOutputName>XXXX.YYYY

where XXXX is the lens plane number (from 0 to NumLensPlanes-1) and
YYYY is a file index from 0 to NumFilesIOInParallel-1.

The galaxy output files are FITS binary tables with the following tags

    index - index which denotes which source galaxy this image is for
            (see below)
    ra - location of image in light in decimal degrees
    dec - location of image in light cone in decimal degrees
    A00 - 00-component of the lensing Jacobian at the image
    A01 - 01-component of the lensing Jacobian at the image
    A10 - 10-component of the lensing Jacobian at the image
    A11 - 11-component of the lensing Jacobian at the image

The lensing Jacobian has its 0-basis vector along the ra-direction and
the 1-basis vector along the dec-direction. The index tag contains a
specially constructed source galaxy index which allows one to use
multiple source galaxy files easily.  It is made in the code like this

    index = fileNum + NumGalFiles*<location in file>

where fileNum is the zero-indexed position of the galaxy source
file in the GalsFileList, NumGalFiles is the total number of files in
the GalsFileList, and <location in file> is the zero-indexed location
of the source galaxy in the fileNum-th source galaxy file.  For
C programmers this construction should be quite familiar. By knowing
how many source galaxy files were input into the code and the index,
one can extract the source galaxy file number and location like this

    fileNum = index mod NumGalFiles
    <location in file> = (index - fileNum)/NumGalFiles

CALCLENS can make lens planes as described above. For making lens
planes, the following options must be set.

    OutputPath - used to output information about lens planes
    NumFilesIOInParallel - controlls I/O loading

    OmegaM - see above
    maxComvDistance - maximum distance to make planes
    NumLensPlanes - number of lens planes
    LensPlanePath - path to output lens planes
    LensPlaneName - base name for lens planes

    rayOrder - see above, not directly used
    bundleOrder - see above, not directly used
    
    LightConeFileList - file containing paths to light cone files
    LightConeFileType - I/O type for light cone file - see
                        lightconeio.c for examples
    LightConeOriginX - X-origin of light in Mpc/h
    LightConeOriginY - Y-origin of light in Mpc/h
    LightConeOriginZ - Z-origin of light in Mpc/h
    LensPlaneOrder - HEALPix order of cells used to spatially index 
         the light cone (usually between 3 and 5)
    memBuffSizeInMB - size of buffer used in memory to sort particles
        (set as big as possible)
    MaxNumLensPlaneInMem - if particles for more than this many lens
         planes are in mem, all planes will be written to disk (set 
         to the number of lens planes usually)
    LightConePartChunkFactor - memBuffSizeInMB/LightConePartChunkFactor 
        extra mem is allocated to each plane as needed when sorting 
        particles (set around 150 or so)
    partMass - particle mass in Msun/h for light cone formats which need it
        (also used as mass of point mass or NFW mass for those tests)
    MassConvFact - conversion factor used to make masses in light cone
        into Msun/h units
    LengthConvFact - conversion factor used to convert distance units
        to Mpc/h
    VelocityConvFact - conversion factor to get velocities into km/s

The velocities are *not* used by the code, but are included so that
indexed light cone format can be used more widely.
