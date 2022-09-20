Cosmology Applications
=======================

About
------

This software package provides tools to simulate gravitational lensing using two different techniques, ray tracing and shear calculation. This software was originally developed as part of my doctoral dissertation (http://thesis.davecoss.com). 

Since I finished my dissertation (May 2010), I have continued to work on this software. In fact, it is still a work in progress, which mostly means adding features. Currently, this includes improving the parallel performance of ray_trace_ellipse and improving the script-ability of physcalc.

Ray Tracing
-----------

ray_trace_ellipse is a program that calculations deflection angles on a grid for light passing may a deflecting mass distribution. Using MPI, ray_trace_ellipse may calculate deflection in parallel across network connected computers, such as cluster.

Shear
-------

Gravitational Lensing Shear is calculated using the relationship of convergence and shear, described by a set of coupled partial differential equations. This is done using the program *physcalc*.

Requires
---------

This software suite requires two libraries that I developed in parallel to this work, which are also available under the GNU GPL.

* libdnstd is a general purpose C++ library that provides various utility functions and classes, generally mathematically focused. This may be found at https://github.com/kd0kfo/libdnstd.git.

* libmygl is a C++ library that provides subroutines and classes to carry out gravitational lensing simulations. This library is located at https://github.com/kd0kfo/libmygl.git.

Additionally, as of version 2.10, this software stored Planes as NetCDF files. This is freely available here: http://www.unidata.ucar.edu/software/netcdf/

*physcalc* requires Flex and Bison. They can be found at http://flex.sourceforge.net/ and http://www.gnu.org/software/bison/ respectively.

Citation
---------

Please cite use of this software or derivatives of this software as:
     Coss, D., "Weak Shear Study of Galaxy Clusters by Simulated Gravitational Lensing.", Ph.D. Dissertation, University of Missouri -- St Louis, 2010.

License
--------

This software is available under the GNU General Public License version 3 (see COPYING).
