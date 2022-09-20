# BINGO: BI-spectra and Non-Gaussianity Operator

The BI-spectra and Non-Gaussianity Operator or, simply, BINGO, is a 
FORTRAN 90 code that numerically evaluates the scalar bi-spectrum and 
the non-Gaussianity parameter fNL in single field inflationary models 
involving the canonical scalar field. 

The code is based on the Maldacena formalism to evaluate the bi-spectrum. 
The code can evaluate all the contributions to the scalar bi-spectrum and 
the non-Gaussianity parameter fNL for an arbitrary triangular configuration 
of the wavenumbers. We should add that, since the perturbations are anyway 
required to be evolved in order to arrive at the bi-spectrum, the code 
can compute the power spectrum too.
 
The methods and the numerical procedures adopted by the code are described in 
detail in the following work: D. K. Hazra, L. Sriramkumar and J. Martin, 
BINGO: A code for the efficient computation of the scalar bi-spectrum, 
JCAP 1305, 026 (2013)]. The results for the arbitrary triangular configurations
of wavenumbers can be found in V. Sreenath, D. K. Hazra and L. Sriramkumar, On 
the scalar consistency relation away from slow roll, In preparation.


## Downloading and installing


1. clone or download bingo-2.0.tar.gz into your home folder and extract it.

2. Open the Makefile

   There are 2 make options (Intel FORTRAN compiler and gfortran (with mpi support). 
   Note that the code has mostly been tested with intel FORTRAN compilers. Though, for other 
   compilers it should work as well). 

3. Choose the correct make option (comment out the inappropriate ones) 
   and save the Makefile.

4. There are six models in the makefile. Choose one model and comment 
   out all the others.
   
   The models available are: 

   a. Quadratic potential with a step (qp-stp)
   b. Small field model with a step (smf-stp)
   c. Axion monodromy model (amm)
   d. Punctuated inflation (pi)
   e. Starobinsky model (Sm)
   f. The power law case (plaw)


5. From a terminal, execute

```
make clean
```

   This will clean the unwanted files (if any) for a fresh install.
  
6. Then compile with 

  ```
  make all
  ```

7. Then run with 

```
make run
```

   The data files will be stored in plots directory. DO NOT DELETE the plots 
   directory.
```   
make prun 
```
   
   runs in 2 nodes. To increase the number of nodes, change the number of 
   nodes in the Makefile.
   
   Only for 2D and 3D plots running in different nodes are allowed. For equilateral 
   and squeezed limits compile with ifort or gfortran instead of mpif90.
   
8. Produce plots with 

```
make figs
```

   You must have GNUPLOT installed to produce them.

9. The following five figure files will be created
 
   a. The behavior of \phi(The field)
      Filename:    phi.eps
   b. The behavior of d\phi/dN (The derivative of the field)
      Filename:    dphi_dn.eps
   c. The behavior of \epsilon (The first slow roll parameter)
      Filename:    epsilon.eps
   d. The scalar power spectrum as a function of the wavenumber
      Filename:    powerspectra.eps  
   e. The f_nl parameter as a function of the wavenumber
      Filename:  f_nl.eps
      This file will contain the f_nl corresponding to a 
      particular term in bi-spectrumb or the total f_nl, 
      which has been selected in fnlparams.ini. This will 
      plot f_nl(k) and holds only for equilateral and squeezed
      configurations. For arbitrary case please refer to 
      point 11, 12 and 14. 

      (For example, if Term=47 in fnlparams.ini is specified the 
      models folder, f_nl.eps will contain f_nl due to the 4th 
      term + f_nl due to the 7th term). TERM = 0 calculates total
      bispectrum from all the terms.
      
      Note that for mpif90 compilation the during the runs 
      the background quantities will not be written in the text files. 
      You can use ifort to compile to generate these files.

10. To run with user specified potential

     a. Create a directory with name my_model in models folder. Copy
        the files from any pre-installed model folder to the my_model 
        folder.
    
     b. In potential.f90 type type your potential which is a function of 
        param_1, param_2, param_3, param_4. You should mention POTENTIAL_USED,
        POTENTIALPRIME and V_PHI_PHI.

        POTENTIAL_USED = The potential [V(\phi)]
        POTENTIALPRIME = d V(\phi)/d \phi [The first  
        derivative with respect to the field)
        V_PHI_PHI= d^2 V(\phi)/d \phi^2 [The second 
        derivative with respect to the field]
    
        In case the potential depends on more than four parameters, you will
        need to define the parameters in the potential.f90 directory. You 
        should change the fnlparams.ini file according to your model.
    
     c. Make the following change in the Makefile. Write, Model = my_model 
        and comment out the other ones.
    
     d. Follow the steps 5,6,7,8.
  
11.  The default code generates f_nl in the triangular region enclosed by 
      0.5 < k2/k1 < 1, and 0 < k3/k1 < 1. However k3/k1 does take a value 
      ~10^(-2) instead of 0. log10(k1) can be fixed in the fnlparams.ini. 

12.  To generate 2D bispectrum shape. 
      
     We have provided 2 plot scripts to plot the 2D bispectrum shapes. 
     Note that for running into several nodes several F_NL files will be 
     generated. In the plots folder $cat *.txt > F_nl_2d.txt shall join 
     them in a single file. $gnuplot > load "2dplot.p" shall generate 
     F_nl_2d.eps file in the plots folder. In gnuplot we do not use 
     interpolation while plotting. A python script is also provided 
     that interpolates F_NL in between points and generate the plot file 
     with the same name in plots folder. To use that use $python plot2D.py.
     
      
      
13.  The code also generates squeezed limit results if specified in the ini file.      
     The squeezed mode is the largest scale mode and is calculated within the code.
  
14.  To generate 3D color plots. 

     Instead of $make all, use 
     ```
     make bingo3d.  
     ```
          
     Change num_k=60 in the corresponding fnlparams.ini file. This will evaluate 
     in 60 * 60 * 60 = 2160000 configurations.
     
     However,
     
     a. Remember to use more nodes since this might take long time to complete. num_k should 
     be integer multiple of the number of nodes.
     
     b. There will be degeneracies in k1, k2, k3 plane. The code is not optimized to 
     reduce this degeneracies since the python plotscript needs the file in a specified format. 
     
     c. For k1, k2,k3 that does not satisfy the triangular configurations, only in this 
     particular case the code replaces them by 0. This is necessary for the plotting purpose.
     
     d. $make gather followed by ./gatherall.out will compile a program to join the 
     files into F_nl_3d.txt. We also provide a python code for plotting. 
     $python plot3D.py shall generate the 3D color plot. Note that you 
     need to have mayavi2 installed in your system. For qp-stp we provide the python plot.
     It plots isosurface for some contour values provided in the script. For other models 
     you need to change the contour values depending on bispectrum shape. In the gather.f90 
     program, the number of files generated (or the number of nodes used) should be provided 
     in total_chains (default set to 60) and the number of lines per file should be provided
     in total_lines.
    
Note that the information about the model and model parameters can be found in the fnlparams.ini files. 


### Vesion history and changelogs :

--------------------------------------------------------------------------------

Vesion 2.0: October 2014 

Bispectrum in arbitrary triangular configurations of wavenumbers can be calculated.

Bispectrum in squeezed configurations to verify the consistency relation away from slow-roll.

Python scripts to plot 2D density plot and 3D contour plots of bispectrum.

Added mpif90 option to run BINGO in multiple nodes.

Term = 0 calculates the total f_nl from all terms in the bispectrum.

--------------------------------------------------------------------------------


Vesion 1.0: February 2013 

The first release of BINGO. Calculation of the bispectrum in the equilateral 
triangular configurations of wavenumbers. 

Calculation of power spectrum.  

--------------------------------------------------------------------------------


In due course, we expect to make a more complete version of the code available.
The code is expected to contain the following features:

1. Tensor bispectrum.

2. NAG library shall be included for improved performance.

3. Interpolation shall be implemented for speeding up the code. 

4. After the Planck 2014 release, we plan to provide an add-on of BINGO for CAMB 
which can directly be used for parameter estimation.

Please write to me (at dhirajhazra@gmail.com) in case you identify a bug in the code 
or if you need any assistance with the code.



for more information, visit https://sites.google.com/site/codecosmo/bingo

This version is: Version 2.0: October 2014
