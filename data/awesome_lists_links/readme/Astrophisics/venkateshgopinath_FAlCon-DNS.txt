**FAlCon-DNS** (Framework of time schemes for direct numerical simulation of annular convection) <br />
This code is written to solve for 2 D convection in an annulus and analyse different time integration schemes.
Contains a suite of IMEX, IMEXRK and RK time integration schemes.
For parallelization, it has OpenMP option. 

It was created from scratch as part of the PhD work at the Geomagnetism Group, Institute de Physique du Globe de Paris (IPGP), Universite de Paris.

The research work using the code is published at: 
V. Gopinath, A. Fournier and T. Gastine, An assessment of implicit-explicit time integrators for the pseudo-spectral approximation of Boussinesq 
thermal convection in an annulus, 460, 110965, JCP, 2022. DOI: https://doi.org/10.1016/j.jcp.2022.110965 

The arxiv version of the published article is available at: https://arxiv.org/pdf/2202.05734.pdf.
Also, the PhD thesis is available at: https://hal.inria.fr/tel-02612607/ 

---------------------------------------
For compiling: <br />

Check the compilers in the Makefile and
go to source directory: <br />
<br />
$ cd src <br />
$ make clean <br />
$ make all <br />
$ cd .. <br />
<br />
For running the code: <br />
<br />
$ mkdir runs <br />
$ cd runs <br />
$ cp ../src/main . <br />
$ cp ../src/input.txt . <br />
<br />
Change the input namelist as per requirement and run: <br />
$ ./main input.txt <br />
<br />
Please modify the Makefile according to your machine requirements. <br />
For questions on usage, please contact: gopinath.venkatesh2@in.bosch.com or venkateshgk.j@gmail.com. 
