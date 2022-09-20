# ECLIPS3D
Public Fortran 90 code for linear wave and circulation calculations, developed originally for planetary atmospheres, with python scripts provided for data analysis. 

Version : March 28, 2019. Developed by Florian Debras, article submitted to A&A. 

This README aims at describing the code and how to use it. Simple examples are provided. It is most easily read directly on the github website, and we are planning to upload a PDF soon. 

4 setups of the code are provided: 2D_axi, 2D_shallow, 3D, 3D_steady. We first describe these setups before explaining how to run the code. 

-------------------------------------------------------
SETUPS OF THE CODE
-------------------------------------------------------
  2D_axi: eigenvector setup in spherical coordinates assuming axisymmetry around the axis of rotation. A longitudinal wavenumber, m, must therefore be provided. 

  2D_shallow: eigenvector setup for shallow water beta-plane. The latitude of the beta plane and characteristic height can be changed.

  3D: eigenvector setup in full 3D, spherical coordinates.

  3D_steady: linear circulation setup, hence matrix inversion. A forcing and a dissipation have to be implemented for a linear steady state to exist. In the current version of ECLIPS3D, the implemented forcing follows Komacek and Showman 2016 and the dissipation is either following Komacek and Showman 2016 or Iro et al. 2005 or is constant through the atmosphere.

-------------------------------------------------------
CONTENT OF THE SETUPS
-------------------------------------------------------

No matter the setup, the directories are then organised as follows:

1) src - contains the source file. The name of the master file depends on the setup (normal_modes_2D_para.F90, normal_modes_shallow.F90, normal_modes_3D.F90 and standing_solver_3D.F90 respectively) but the other files are the same, except for the steady state solution.
Namely: - mod_data.F90: contains the data public to all modules of the code.
        - mod_init_para.F90: initialises MPI
        - mod_init_matrix.F90: distributes the matrix on the processors
        - mod_fill_matrix.F90: fills the matrix with the equations
        - mod_eigenvalues.F90: calculates eigenvalues and eigenvectors of the matrix. Does not exist for 3D_steady, instead mod_solver.F90 inverts the matric to obtain the steady state linear circulation.

Two other files allow to select and write the eigenvectors: study_eigenvectors.F90 read the eigenvector file and select some according to the selection procedures described in Debras et al. 2019. write_eigenvectors.F90 write the selected eigenvectors in a new file to feed them into python. NOTE: in the 3D_steady version, both these files are contained in read_solution.F90, that just allows to write the steady solution in a format suited for the given python file.

2) bin - contains the Makefile, and the intermediate compilation files. 

3) run - contains the input files and the exe file created by bin. 

4) data - contains the input data to launch the code as well as the output.

5) python - contains typical python files to generate an initial set of data and study the output. We provide simple example in these python files, namely axisymmetric state at rest and an initial, steady baroclinically unstable jet. For the steady linear circulations, the provided example follows Komacek & Showman 2016 for the heating rate and radiative and drag timescales. 

-------------------------------------------------------
OVERVIEW OF THE CODE
-------------------------------------------------------

ECLIPS3D first initialises the parallel computing, performed by MPI, and allocates the memory on each processor to store the matrix to invert/for eigenvector calculation. The size on each processor is set by the parameter nb, and the performances of the program are very dependent to nb. It seems that nb ~ ntot/sqrt(n_proc)/10 is a reasonable number for good performances. 

Once initialised, the program calls mod_fill_matrix in order to calculate the coefficients of the matrix. In this version, we have decided to generate input data file from python (which deals with interpolations very easily) and read them afterwards in  fortran in ECLIPS3D. Changing that is no problem.

We then calculate the coordinates of the grid, the drag and radiative timescales if needed (a tentative at diffusion is also proposed, although not tested yet), the sound speed and brunt vaisala frequency. We then write an output file with the initial state so that it can be used to reconstruct dimensional variables once the program has run. 

Afterwards, we fill the matrix. There is a loop on ntot which set to one only the ith point and zero all the other one, followed by an inner loop on the coefficient of the matrix so that the coefficient i,j is the impact on the jth point from the ith point only. The beginning of the loop ensures that some points are always zero, according to boundary condition. The default setup assumes north south symmetry, hence v must be zero at the equator. The usual condition on v is that vcosphi must be zero at the pole, hence we set to zero all the terms that do not cancel in the equation of v when multiplied by cos phi and setting cos phi = 0. 

Finally, the last routine either inverts the matrix, calculates the full spectrum of eigenvectors or simply calculates the selected eigenvectors. This involves combining numerous sclapack routine, as described in the submitted paper. 


**IMPORTANT NOTE**: In this version, all the matrices are filled with complex quantities. This is inspired by the 2D, axisymmetric case with an initial state at rest where there is a difference of phase of Pi between u and v, hence one is pure imaginary when the other is real. THIS HAS TO BE CHANGED, as keeping a complex matrix increases the size of the memory to store the matrix, whereas the real part of each number is always zero in the 3D version. The interest lies in the fact that reading the complex eigenvectors of a real matrix is different than just reading the eigenvector of a complex matrix in SCALAPACK, and we have not implemented the former case in python.                                    
                                        
-------------------------------------------------------
RUNNING ECLIPS3D
-------------------------------------------------------

In this section we detail how to run ECLIPS3D, with the test cases provided. 


##########################

**The most important thing is to configure your Makefile properly to ensure that the code loads the BLAS, LAPACK and SCALAPACK libraries properly.** 

Typically, if you are using MKL, you should add in the Flags : 

"FLAGS = -I${MKLROOT}/include/intel64/lp64 -I${MKLROOT}/include"

and in the libraries something of this kind : 

"LIBS =  ${MKLROOT}/lib/intel64/libmkl_lapack95_lp64.a -L${MKLROOT}/lib/intel64 -lmkl_scalapack_lp64 \
        -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 -liomp5 -lpthread -lm -ldl -lblas -lcurl"

If you never heard of Lapack and Scalapack before, the easiest thing might be to install LAPACK (e.g, https://pheiter.wordpress.com/2012/09/04/howto-installing-lapack-and-blas-on-mac-os/) and then scalapack from the netlib website, following the installation guide (http://www.netlib.org/lapack/lawnspdf/lawn93.pdf).

Then, ECLIPS3D should work if you just change the libs line in the makefile :
"LIBS = /usr/local/lib/libblas.a /usr/local/lib/liblapack.a /usr/local/lib/libscalapack.a"

##########################

Assuming that compilation works fine, here is a detail of how to run the code for the three eigenvectors setup, and afterwards  for the steady circulation. The 'selected eigenvectors' is not yet implemented in the default version of the code, but should be ASAP and can be given on demand. 

1) In the python repertory: run the script data_to_ECLIPS3D.py to generate an atmosphere initialised at rest. You need to change the 'output_dire' at the beginning of the file to the correct path to your computer. Choose a number of radial (Nz), latitudinal (Nlat) (and longitudinal (Nlong) if you are running the 3D code) points, according to your number of processors. Resolution and execution time are alluded to in the paper.  

2) In the bin directory, type 'make' (see above for discussions about libraries). You might need to adapt to other compilers if mpif90 is not on your computer. This is easily done in the Makefile.

3) In the run directory: open data.input and change the directory, adapt it to the planet you are considering, and the number of points you are using. For low number of points, nb=255 seems to always be a good choice for shortening execution time. For higher resolution runs, nb = ntot/sqrt(nprocs)/10 seems adequate. Globally, documentation is missing on that point in SCALAPACK.

4) Open timescales.input. If you want free waves, set both these values to zero. If you want waves with a linear dissipation, such is performed in the paper on the superrotation of hot Jupiters that will be submitted 21st of April, choose the characteristic timescales you want. More documentation is provided in Iro et al. 2005 and Komacek and Showman 2016.

5) Run the program (the exe file depends on the setup) : mpirun - np #NBPROCS ./ECLIPS3D.exe data.input timescales.input  
Different things should be printed:
                  - the name of the parameter file: "parameter file name=data.input"
                  - the local leading dimension, a useful information about the size of the matrix. 
                  - Then "all read", "timescales ok" telling that the initialisation is going well. 
                  - "Begin eigenvalues finder" tells you that the matrix is ready
                  - Then a few steps in the calculation, beginning by "Diagonal killed", preparing for Schur decomposition
                  - The scary "last routine", which means that the most time consuming routine is for now
                  - And finally "job finished" when everything is over.
                  
If you get a message of the form "PZGEHRD failure", it means that the scalapack routine PZGEHRD had a problem. I can't help you from there, probably something wrong with your installation/setup. 

6) Now, a very heavy file has been written in the data repertory, containing all the eigenvectors. This might not be the most convenient way, but it has a few advantages I did not want to give up on. The rest is handled by study_eigenvectors and read_eigenvectors. Study_eigenvectors reads the pressure of the modes, checks whether they are numerically correct, select them according to given criteria (in the default version, less than two zeros in every coordinates) and then writes the number of the modes in a file for write to read it, and write a final, light output file. In my todo list, I need to make automatic the fact that study_eigenvectors and write_eigenvectors know the directory and the number of points. For now, you need to change it manually in the fortran files (except in the 3D version for the number of points). So type "gfortran ../src/study_eigenvectors.F90 -o study.exe" and then "./study.exe . A list of information, namely the number, frequency, growth rate (and number of zeros in the 3D version) will be printed on your terminal.

7) Do the same for write_eigenvectors: change the directory and number of points, then type 
"gfortran ../src/write_eigenvectors.F90 -o write.exe" and then "./write.exe" The frequency of the first modes and the number of the modes will be printed on screen, with the last one being always zero. You now have an output file in the data repertory called "selected_eigenvectors.dat" that contains everything you need !

8) Last step is to visualize your results with python. Open the plot_output.py file.

Three things are needed: the directory, the rho_cs_ns.dat file and the selected_modes.dat file, created by write_eigenvectors.F90. Note: the way to read the file is old school, because I had some python-Fortran problems in the beginning. This can easily be improved. 

There are many displyaing options in this python file, the default one being a window with all 5 perturbed variables (normalized to their energy contribution, see Thuburn et al.2002). "n_auto" modes will be displayed, the first one being defined for example at line 141 in the 2D axisymmetric version (here, 0 : "t>0".) You can interpolate to have higher resolution image, as well as choose to plot in the longitudinal direction given a 'm' value is specified. If you are in 3D, you get even more choice. These pyhton files needs additional commenting which will be given in the near future. Again, help can be given on request. 




In the steady circulation setup (3D_steady repository), simpply do the same up to step 6. The outputs on screen will be slightly different, and note that the initial python file does not create an atmosphere at rest only, but adds a heating function taken from Komacek & Showman 2016, with the addition of an exponential damping in the high atlosphere (in order to mimic the sponge layer). Step 7: simply call the routine read_solution.F90, which sole purpose is to write the data in an appropriate way. Then, visualize your only output with plot_output.py. There is only one output as this is only a matrix inversion you are performing. 


-------------------------------------------------------
LAST WORD
-------------------------------------------------------       

This version of ECLIPS3D has been adapted to be used as simply as possible, although the original program was not developed in a user friendly way. There may still be some places in the code which prints weird outputs (like 'lol', my usual debugging output) or which are not easy to understand. Do not hesitate to contact me with any request about ECLIPS3D. 
