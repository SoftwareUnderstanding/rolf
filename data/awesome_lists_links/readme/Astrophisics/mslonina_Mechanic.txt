Mechanic
--------

### Overview

We develop the Mechanic package, which is a flexible numerical framework to
handle and automate massive numerical simulations. We assume that these computations rely
on testing a huge range of initial conditions, while each test (single simulation run) may
be computed as a standalone task, either on one or a group of CPUs. A natural way to
analyse such data sets is to run the simulations in parallel; however, managing that "by
hand" is usually a cumbersome job and introduces human-based errors. A usage of queue
scripts and job control mechanisms remains still task-dependent.

The Mechanic framework relies on the core-module approach (similar to
server-client architecture, in terms of the MPI farm model). It provides a
relatively abstract layer of flexible and powerful module interface which
makes it possible to easily adapt the existing code. The modules are loaded
dynamically during run-time. The Mechanic module interface provides 
a template structure that allows to run specific functions on particular nodes, e.g. only
on slave CPUs. User may choose among a number of  available functions to communicate the
slave tasks with the Mechanic core, and manage the simulation in every detail. The module
interface applies both to C and Fortran2003 codes. It is possible to connect software
written in other programming languages/frameworks, i.e. NVIDIA CUDA or OpenCL.

The underlying idea of the Mechanic's core is to make it completely
transparent to the numerical problem handled by the given, external (user
supplied) module. Thus, the framework might be installed system-wide and
become a helper tool for users, who need to perform a large number of
computations. The top layer of the framework is focused on handling 
massive simulations in every technical aspect -- through basic 
configuration of the runs, sending, receiving and storing data, restarting 
runs at each stage.

The code is written in C, uses HDF5 data layer and comes with Fortran
bindings. It was developed and tested to work equally well in fake-MPI mode
(like on a single CPU) and on a large CPU cluster, either in 32- and 64-bits environments
(currently, Linux and Mac OS X are actively maintained). 

Although the framework remains in early alpha stage, some existing Fortran77 and C codes
were successfully ported and ran with Mechanic, showing huge potential of the code, as
well as number of features to improve and develop. The development and testing the
framework is ongoing.

Mechanic is BSD-licensed. The source code package comes with few example
modules and is freely available at http://git.astri.umk.pl/projects/mechanic

### Publications

- Slonina M., Gozdziewski K., Migaszewski C., 2012arXiv1202.6513S
- Migaszewski C., Slonina M., Gozdziewski K., 2012arXiv1205.0822M
- Gozdziewski K. et al, 2012arXiv1205.4164G
- Slonina M., Gozdziewski K., Migaszewski C., Rozenkiewicz A., 2012arXiv1205.1341S
- Slonina M., Gozdziewski K., Migaszewski C., Astrophysics Source Code Library, record ascl:1205.001

### Posters

- Slonina M., Gozdziewski K., Migaszewski C., Simtech2011 (Stuttgart, June 2011)
- Slonina M., Gozdziewski K., Migaszewski C., Orbital Couples: "Pas de Deux" in the Solar System and the Milky Way (Paris, October 2011)

### Acknowledgments

This project is supported by the Polish Ministry of Science and Higher Education through the grant N/N203/402739. This work is conducted within the POWIEW project of the European Regional Development Fund in Innovative Economy Programme POIG.02.03.00-00-018/08.
