```
        ,----,                                                                  
      ,/   .`|                                                                  
    ,`   .'  :                             ,---,.                               
  ;    ;     /                           ,'  .' |                               
.'___,/    ,' __  ,-.                  ,---.'   |  __  ,-.   ,---.              
|    :     |,' ,'/ /|                  |   |   .',' ,'/ /|  '   ,'\   ,----._,.
;    |.';  ;'  | |' | ,---.     ,---.  :   :  :  '  | |' | /   /   | /   /  ' /
`----'  |  ||  |   ,'/     \   /     \ :   |  |-,|  |   ,'.   ; ,. :|   :     |
    '   :  ;'  :  / /    /  | /    /  ||   :  ;/|'  :  /  '   | |: :|   | .\  .
    |   |  '|  | ' .    ' / |.    ' / ||   |   .'|  | '   '   | .; :.   ; ';  |
    '   :  |;  : | '   ;   /|'   ;   /|'   :  '  ;  : |   |   :    |'   .   . |
    ;   |.' |  , ; '   |  / |'   |  / ||   |  |  |  , ;    \   \  /  `---`-'| |
    '---'    ---'  |   :    ||   :    ||   :  \   ---'      `----'   .'__/\_: |
                    \   \  /  \   \  / |   | ,'                      |   :    :
                     `----'    `----'  `----'                         \   \  /  
                                                                       `--`-'   
```

![alt text](https://github.com/pelahi/TreeFrog/blob/master/treefrog.png)

# TreeFrog (formerly HaloTree)

================================================================================================

 ## developed by:
    Pascal Jahan Elahi (continuously)
    Additional contributors:
    Rhys Poulton
    Rodrigo Tobar

================================================================================================

## Content
    (for more information type make doc in main dir and in NBodylib dir and
    see documents in the doc directory)

    src/        contains main source code for the algorithm
    doc/        contains Doxygen generated latex and html file of code
    NBodylib/   submodule: contains library of objects and routines used by algorithm (can also be used by
                other routines)
    tools/      submodule: contains python tools of manipulating/reading output

================================================================================================

 ## Compiling (see documentation for more information)

    TreeFrog uses CMake as its build tool. cmake is used to perform system-level checks,
    like looking for libraries and setting up the rules for the build, and then generates the
    actual build scripts in one of the supported build systems. Among other things, cmake supports
    out-of-tree builds (useful to keep more than one build with different settings, and to avoid
    cluttering the original source code directories) and several build system, like make and
    ninja files.

    TreeFrog uses submodules so if you have a fresh clone use

    git submodule update --init --recursive

    to update the submodules use

    git submodule update --recursive --remote

    The simplest way of building is, standing on the root your repository, run cmake to produce
    Makefiles and then compile with these steps:

    mkdir build
    cd build
    cmake .. # By default will generate Makefiles
    make all

    There are a variety of options that can be invoked
    and these can be viewed using
    cmake -LH
    (though this only works after having run cmake at least once)

    Although documentation is present on the readthedocs site, extra documentation can be produced
    by typing
    make doc
    which will produce html and latex documents using Doxygen. This will be located in
    doc/html/index.html
    and
    doc/latex/refman.tex

    Note that TreeFrog and all variants do not support non-Unix environments. (Mac OS X is fine; Windows is not).

================================================================================================

## Running (see documentation for more information)

    This is a MPI+OpenMP code that reads in particle IDs information between various structure
    catalogues and cross matches catalogues assuming that particle IDs are unique and constant
    across snapshots. Though it is built as a cross correlator (in that it can match particles
    across several different catalogues), its principle use is as halo merger tree builder. The
    code produces links between objects found at different snapshots (or catalogues) and uses
    several possible functions to evaluate the merit of a link between one object at a given
    snapshot (or in a given catalogue) to another object in a previous snapshot
    (or different catalogue). It can also produce a full graph.

    This code naturally reads VELOCIraptor output and is optimised for it but can also read output
    from other structure finders like AHF.

    Running the code is
    mpirun -np numberofmpi ./bin/treefrog -i filelist.txt -o outname -s numberofsnapsorcatalogues

    The code also has many other command line arguments. Simply pass -?

    Note that building a tree can be quite memory intensive for large simulations with lots of snapshots.
    In mpi mode, the snapshots are split so as to approximately have the same number of particles (or halos) in
    structures per mpi process (load balance the memory footprint). This means that some mpi threads will
    process significantly more snapshots than others (consider early times where few particles belong
    to groups compared to late times where lots of structure has formed.) In that case it may be useful
    to play with the load balancing when running in mpi. This can be done using a single mpi thread and
    specifying the desired number of mpithreads and the desired number of particles per mpi thread.
    mpirun -np 1 ./bin/treefrog -i filelist.txt -o outname -s numberofsnapsorcatalogues -z nummpi -n numpermpi
    This will produce a file containing the load balance (ie: what files a mpi process should read).

================================================================================================

## Tools:

    Contains some example of reading routines for velociraptor output. For example will show
    how a routine will read the output of velociraptor.
