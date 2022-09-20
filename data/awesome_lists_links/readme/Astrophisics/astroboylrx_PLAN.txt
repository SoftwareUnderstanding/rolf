# PLAN

[![ASCL](https://img.shields.io/static/v1?label=ASCL&message=1911.001&color=brightgreen&link=https://ascl.net/1911.001)](https://ascl.net/1911.001)
[![ADS](https://img.shields.io/static/v1?label=ADS&message=MethodPaper&color=blue&link=https://ui.adsabs.harvard.edu/abs/2019ApJ...885...69L/abstract)](https://ui.adsabs.harvard.edu/abs/2019ApJ...885...69L/abstract)
[![Zenodo](https://zenodo.org/badge/53685063.svg)](https://zenodo.org/badge/latestdoi/53685063)

PLanetesimal ANalyzer (`PLAN`, [Li et al. (2019)](https://doi.org/10.3847/1538-4357/ab480d)) identifies and characterizes planetesimals produced in numerical simulations of the Streaming Instability ([Youdin & Goodman 2005](https://doi.org/10.1086/426895)) that includes particle self-gravity with code [`Athena`](https://github.com/PrincetonUniversity/Athena-Cversion) ([Stone et al. 2008](https://doi.org/10.1086/588755), [Bai & Stone 2010](https://doi.org/10.1088/0067-0049/190/2/297), [Simon et al. 2016](https://doi.org/10.3847/0004-637X/822/1/55)).  `PLAN`  has already been used in the analyses of [Li et al. (2018)](https://doi.org/10.3847/1538-4357/aaca99), [Abod et al. (2018)](https://doi.org/10.3847/1538-4357/ab40a3), and [Nesvorný et al. (2019)](https://doi.org/10.1038/s41550-019-0806-z) (featured on the [cover](https://www.nature.com/natastron/volumes/3/issues/9) of *Nature Astronomy*), and more studies in progress.

Currently, `PLAN` works with the 3D particle output of `Athena` and finds gravitationally bound clumps robustly and efficiently.  `PLAN` — written in `C++` with `OpenMP/MPI` — is massively parallelized, memory-efficient, and scalable to analyze billions of particles and multiple snapshots simultaneously.  The approach of `PLAN` is based on the dark matter halo finder `HOP` ([Eisenstein & Hut 1998](https://doi.org/10.1086/305535)), but with many customizations for planetesimal formation.  `PLAN` can be easily adapted to analyze other object formation simulations that use Lagrangian particles (e.g., `Athena++` simulations). PLAN is also equipped with a toolkit to analyze the grid-based hydro data (`VTK` dumps of primitive variables) from Athena, which requires Boost MultiDimensional Array Library.

## Demo

The picture below is a snapshot of the solid surface density from one of our high-resolution shearing box simulations (of the coupled gas-dust system in a local patch of protoplanetary disks). Self-bound clumps have already formed from gravitationally collapse in this snapshot. All of the clumps identified by PLAN are marked by white circles that illustrate their Hill spheres.

![](Demo4Readme.jpg)

## Compile & Run
CMake is needed to generate a Makefile and compile this program. Boost headers are also required. `PLAN` can be accelerated with `MPI` and `OpenMP`.

You may just run `cmake` and `make` to build `PLAN`.

```shell
➜  PLAN $ ls
CMakeLists.txt README.md      src
➜  PLAN $ mkdir build && cd ./build
➜  build $ cmake ..
-- The C compiler identification is AppleClang 9.0.0.9000039
-- The CXX compiler identification is AppleClang 9.0.0.9000039
-- Check for working C compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc
-- Check for working C compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Check for working CXX compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++
-- Check for working CXX compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
============================================================
Run cmake without any option will build a serial program.
The availabel build options are:
    -DPARALLEL=ON   to enable MPI
    -DOPENMP=ON     to enable OpenMP
    -DHYBRID=ON     to enable MPI+OpenMP
N.B.: the chosen option is cached after the first use.
To switch, add -DOPTION=OFF to turn off the previous choice.
eg. cmake -DHYBRID=OFF -DOPENMP=ON ../
You can always delete cache files/folders for a fresh start.
PS: clang/gcc built-in with macOS does not support OpenMP.
============================================================
============================================================
Now, generating Makefile for serial program...
============================================================
-- Boost version: 1.66.0
-- Configuring done
-- Generating done
-- Build files have been written to: /Users/rixin/PLAN/build
➜  build $ make -j 4
Scanning dependencies of target plan
[ 60%] Building CXX object CMakeFiles/plan.dir/src/global.cpp.o
[ 60%] Building CXX object CMakeFiles/plan.dir/src/analyses.cpp.o
[ 60%] Building CXX object CMakeFiles/plan.dir/src/tree.cpp.o
[ 80%] Building CXX object CMakeFiles/plan.dir/src/main.cpp.o
[100%] Linking CXX executable plan
[100%] Built target plan
➜  build $ ls
CMakeCache.txt      CMakeFiles          Makefile            cmake_install.cmake plan
```

You can specify the environment variables `CC` and `CXX` to tell cmake which compilers to use. In addition, three options are available to build `PLAN` with MPI and/or OpenMP.

```shell
➜  build $ rm -rf ./*
zsh: sure you want to delete all 5 files in /Users/rixin/PLAN/build/. [yn]? y
➜  build $ export CC=gcc; export CXX=g++
➜  build $ cmake -DHYBRID=ON ..
-- The C compiler identification is GNU 7.3.0
-- The CXX compiler identification is GNU 7.3.0
-- Checking whether C compiler has -isysroot
-- Checking whether C compiler has -isysroot - yes
-- Checking whether C compiler supports OSX deployment target flag
-- Checking whether C compiler supports OSX deployment target flag - yes
-- Check for working C compiler: /opt/local/bin/gcc
-- Check for working C compiler: /opt/local/bin/gcc -- works
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Detecting C compile features
-- Detecting C compile features - done
-- Checking whether CXX compiler has -isysroot
-- Checking whether CXX compiler has -isysroot - yes
-- Checking whether CXX compiler supports OSX deployment target flag
-- Checking whether CXX compiler supports OSX deployment target flag - yes
-- Check for working CXX compiler: /opt/local/bin/g++
-- Check for working CXX compiler: /opt/local/bin/g++ -- works
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Detecting CXX compile features
-- Detecting CXX compile features - done
============================================================
Run cmake without any option will build a serial program.
The availabel build options are:
    -DPARALLEL=ON   to enable MPI
    -DOPENMP=ON     to enable OpenMP
    -DHYBRID=ON     to enable MPI+OpenMP
N.B.: the chosen option is cached after the first use.
To switch, add -DOPTION=OFF to turn off the previous choice.
eg. cmake -DHYBRID=OFF -DOPENMP=ON ../
You can always delete cache files/folders for a fresh start.
PS: clang/gcc built-in with macOS does not support OpenMP.
============================================================
============================================================
Now, generating Makefile for hybrid program (MPI+OpenMP)
============================================================
-- Found MPI_C: /opt/local/lib/mpich-gcc7/libmpi.dylib (found version "3.1")
-- Found MPI_CXX: /opt/local/lib/mpich-gcc7/libmpicxx.dylib (found version "3.1")
-- Found MPI: TRUE (found version "3.1")
-- Found OpenMP_C: -fopenmp (found version "4.5")
-- Found OpenMP_CXX: -fopenmp (found version "4.5")
-- Found OpenMP: TRUE (found version "4.5")
-- Boost version: 1.66.0
-- Configuring done
-- Generating done
-- Build files have been written to: /Users/rixin/PLAN/build
➜  build $ make -j 4
Scanning dependencies of target plan
[ 60%] Building CXX object CMakeFiles/plan.dir/src/analyses.cpp.o
[ 60%] Building CXX object CMakeFiles/plan.dir/src/global.cpp.o
[ 60%] Building CXX object CMakeFiles/plan.dir/src/tree.cpp.o
[ 80%] Building CXX object CMakeFiles/plan.dir/src/main.cpp.o
[100%] Linking CXX executable plan
[100%] Built target plan
➜  build $ ls
CMakeCache.txt      CMakeFiles          Makefile            cmake_install.cmake plan
```

To clean CMake results and cache, just delete the `build` directory or its contents.

---

While analyzing real data, `PLAN` needs to calculate the mass of each particle as well as all the density thresholds, which requires extra parameter information beyond the particle data dumps.  Thus, `PLAN` usually takes an input file (see the example file in the `scripts` folder) that specifies the hydro-grid resolution, the dust-to-gas surface density ratio (or metallicity), and the relative strength of particle self-gravity (see Eq. 8 in [Li et al. (2019)](https://arxiv.org/abs/1906.09261)) used in simulations.  Below shows the output of an example run.

```shell
➜  build $ ./plan
Program begins now (local time: Thu Oct 31 23:49:40 2019).
*******************************************************************************
USAGE:
./plan -c <num_cpus> -i <data_dir> -b <basename> -p <postname>  -f <range(f1:f2)|range_step(f1:f2:step)> -o <output> [-t <input_file_for_constants> -s 10 -x -0.1,0.1 -y -0.05,0.05 --flags]
Example: ./plan -c 64 -i ./bin/ -b Par_Strat3d -p ds -f 170:227 -o result.txt -t plan_input.txt --Verbose --Find_Clumps
[...] means optional arguments. Available flags:
Use --Help to obtain this usage information
Use --Verbose to obtain more output during execution
Use --Debug to obtain all possible output during execution
Use --Combined to deal with combined lis files (from all processors)
Use --Find_Clumps to run clump finding functions
Use --Save_Clumps to save all clumps to particle lists
        (use '-s N' to sub-sample (1 in N) the particle list for outputting to save storage)
Use --No_Ghost to skip making ghost particles
Use --Basic_Analyses to perform basic analyses, which will output max($\rho_p$) and $H_p$
Use --Density_Vs_Scale to calculate max($\rho_p$) as a function of length scales
Use --Temp_Calculation to do temporary calculations in TempCalculation()
If you don't specify any flags, then --Find_Clumps will be turned on automatically.
➜  build $ mpirun -np 2 ./plan -c 256 -i ../data/ -b Par_Strat3d -p combined -f 77:80:3 -o ./result.txt
Program begins now (local time: Fri Nov  1 00:02:39 2019).
*******************************************************************************
Set the number of available threads for OpenMP to 6. This number can also be fixed manually by specifying "num_threads" in the parameter input file. 
Note that every processor in MPI will utilize such number of threads in its own node. It is recommendeded to use --map-by ppr:n:node in the Hybrid scheme. 
For example, to obtain the best performance, if there are 16 cores per node, then
	mpirun -np XX --map-by ppr:2:node:pe=8 ./your_program ...
with num_threads=8 will initialize 2 processors in each node and each processor will utilize 8 threads in the OpenMP sections. In this way, the entire node is fully utilized.
Processor 1: Finish clump finding for t = 38.5, found 286 clumps;  Mp_max = 4.59336e-05, Mp_tot = 0.00130005(51.864338%) in code units.
Processor 0: Finish clump finding for t = 40, found 252 clumps;  Mp_max = 9.94499e-05, Mp_tot = 0.00157491(62.829971%) in code units.
Max waiting time among all processors due to Barrier(): 4.6386e-05s.
*******************************************************************************
Program ends now (local time: Fri Nov  1 00:03:05 2019). Elapsed time: 2.604206e+01 seconds.
```


