<a name="top"></a>

# MG-MAMPOSSt, a code to test gravity with the mass profiles of galaxy clusters 

- [Go to Main page](https://github.com/Pizzuti92/MG-MAMPOSSt)
- [Introduction](#Introduction)
- [Documentation](#Documentation)
- [Install](#Install)
- [Overview of usage](#Overview-of-usage)
- [Test and Tutorial](#Test-and-Tutorial)
- [Extension and Update](#Extension-and-Update)
## Introduction

MG-MAMPOSSt is a FORTRAN code that extended the MAMPOSSt algorithm of G. Mamon, A. Biviano and G. Boué - 
which performs Bayesian fits of models of mass and velocity anisotropy profiles to the distribution of tracers in projected phase space -
to handle modified gravity models and constrain their parameter space. The new version implements two distinct types of gravity modifications, 
namely general chameleon (including $`f(\mathcal{R})`$ models) and beyond Horndeski gravity (Vainshtein screening). MG-MAMPOSSt efficently explores the parameter space either by computing the likelihood over a multi-dimensional grid of points or by performing a simple Metropolis-Hastings MCMC. 
The code requires a Fortran90 compiler or higher and makes use of the Python3 [getdist package](https://github.com/cmbant/getdist) of Antony Lewis to plot the marginalized distributions in the MCMC mode.

### Documentation

MG-MAMPOSSt has been developed by L. Pizzuti, I.D. Saltas G. Mamon L. Amendola and A. Biviano from the MAMPOSSt version of A. Biviano, with a precious contribution by S. Sartor and S. Sanseverinati. A full description of the code functionalities, input parameters and output files is given in the [documentation](https://github.com/Pizzuti92/MG-MAMPOSSt/blob/main/Documentation.pdf), while the functions/subroutines available esternally are further described in the [API docmumentation](https://github.com/Pizzuti92/MG-MAMPOSSt/blob/main/API%20Documentation.pdf). To test some of the subroutines, refer to the source code testMAM.f, which can be compiled and executed as
```bash
 f95 -o testMAM.e testMAM.f -L build/ -lMAM -L build/GamI/ -lGamI -L build/Newuoa/ -lNewuoa -L build/JJin/ -lJJin -L build/Utili/ -lUtili -L build/Powell/ -lPowell
 ./testMAM.e
``` 
Details of the original MAMPOSSt method can be found in [Mamon et al., 2013](https://academic.oup.com/mnras/article/429/4/3079/1012297?login=false). The updated version of the (GR) MAMPOSSt code can be downloaded [here](https://gitlab.com/gmamon/MAMPOSSt). 


---

## Install

To install and run MG-MAMPOSSt, download the .zip file which contains all the dependencies. Note that the main source code gomamposstoptS.f requires additional routines which are stored in the various folders shipped within the code. Some routines are taken from free FORTRAN libraries available on the web. Credits to the developers are given in the header of the source files of these routines.

### Install with CMake

The configuration and installation requires at least CMake 3.17.1. 
In the working directory, execute the following commands:
```bash
$ mkdir build
$ cd build/
$ cmake ..
$ cmake --build .
$ sudo cmake --install .
$ cd ..
```
To run and test the code, execute:
```bash
$ gomamposstopt < gomamposst_x.inp
```
which produces the main outputs, or
```bash
$ ./script/script_runmam.sh  
```
which also generates additional plots if the MCMC mode is selected (see below).
Note that to run the above script, permissions should be changed to make it executable. Otherwise, one can simply use the ``` sh ``` environment.

## Overview of usage

Here we summarize all the necessary information to perform a complete run of the code.

### Dataset

The directory data/ stores the datafiles of projected phase spaces (p.p.s) that serve as input to the MG-MAMPOSSt procedure. The files are structured as a table where the number of rows coincides with the number of data points. The first column is the projected radius in units of ```kpc```, the second and thirds columns represent the l.o.s. velocities and the associated errors in units of ```km/s```. Note that data points should be given in the rest frame of the cluster.
The first two lines are considered as comment lines when MG-MAMPOSSt read the data.
If a fourth column is given, the numbers should correspond to the weights associated to each measurement. Set the parameter **weights=1** in the input parameter file (see [next section](#input-parameters)), otherwise the weights will be ignored.

The test data file provided corresponds to a projected phase space realization of a relaxed dark matter halo, populated with a NFW model characterized by r_200=1.41 Mpc and r_s=r_nu=0.33 Mpc. As for the velocity anisotropy profile, the halo is generated with a Tiret model (implemented in the MG-MAMPOSSt code) with a parameter beta=1.41. 

### Input parameters
Input parameters are stored in the file input_pars/pars_test.txt. All the parameters must be given in a format **"label = "value"**. The "label"s are mandatory while the "value"s, if not given, are set by default, except for the guess values of the free parameters. The parameters can be  **integers*4**, **reals*8** or **string** and they are divided in four main groups:

* **Number of iterations (integers):** for the grid search mode, they represent the number of points in each free parameter over which the likelihood is computed. Note that, if  **number of iterations** is even, the number of grid points will be **number of iterations+1** in order to have an even number of grid sample below and above the guess value.  If set to 0 or 1, the parameter is fixed to its guess value, except for specific cases. When MG-MAMPOSSt is in MCMC mode, if **number of iterations** is different from zero, then the corresponding parameters are optimized within the chain. Otherwise the parameters are fixed to the guess value. Default value is zero.
   - **nr200**: Number of iteration for the virial radius.

   - **nrc**: Number of iteration for the tracers scale radius r_nu. For **nrnu=-1**  assumes that r_nu is equal to the scale radius of the mass profile r_s in the fit (option: "Light Follows Mass"). If **nrnu=-2** the code first fits the number density profile alone and the best fit found is then used as fixed value in the MG-MAMPOSSt procedure. 
   - **nrs**: Number of iteration for the mass profile scale radius.  For **nrs-1**  assumes that r_s is equal to the scale radius of the number density profile r_nu in the fit (option: "Mass Follows Light"). If **nrs=-2** the mass scale radius is computed by using the theoretical relation of Macciò et al., 2008 (LambdaCDM option).
   - **nbeta**: Number of iteration for the anisotropy parameter. If **nbeta=-1** the anisotropy profile is forced to be a Mamon&Lokas profile with beta=r_beta=r_s. If **nbeta=-2** the Hansen-Moore model is assumed (beta(r) related to the matter density rho_m(r)).
   - **nA1**: Number of iteration for the first MG parameter.
   - **nA2**: Number of iteration for the second MG parameter. For the case of general chameleon gravity (**M(r)=mNFW_GC**, see below), A2 corresponds to the coupling constant **Q**. In this case, if **nA2=-1** it forces the case of $f(R)$ gravity (Q=1/\sqrt{6}).
   - **nbeta2**: Number of iteration for the second velocity anisotropy parameter. It works only if the selected beta model is "gT" or "gOM"
   
* **Guess values (reals):** they serve as an initial guess for the MG-MAMPOSSt fit both in grid and MCMC mode. If the corresponding number of iterations is set to zero, or one when in the grid-search option, the parameter guess is kept fixed within the code.
   * **r200g**: guess starting value of the characteristic "virial" radius for the mass profile, measured in units of Mpc. 

    * **rcg**: guess starting vaule for the scale radius of the number density profile,  units of Mpc. 
    
    * **rsg**: guess starting value for the scale radius of the mass profile, in  units of Mpc. 

    * **betag**: starting guess for the velocity anisotropy profile parameter. If the selected model is "C", "T" or "O", the parameter is dimensionless and it is given by the quantity (1-beta_inf)^(-1/2), where beta_inf correspond to the value of the anisotropy at infinity. **betag=1** indicates fully isotropic orbits (beta_inf=0). For the other anisotropy models **betag** should be given in units of Mpc.
    
    * **A1g**: Initial guess for the first modified gravity parameter. For **M(r)=mNFW_GC** the parameter is in unit of 1e-5. For the case of model **gNFW**, **A1=gamma** is the free exponent which characterizes the profile.
    
    * **A2g**: Initial guess for the second modified gravity parameter.
    
    * **beta2g**: Initial guess for the second velocity anisotropy parameter (always dimensionless) and corresponding to (1-beta_0)^(-1/2), where beta_0 correspond to the value of the anisotropy at r=0.
    
* **Model options (reals/integers/strings):** this family of values allow to select cosmological environment such as the value of the  Hubble parameter and the average redshift of the cluster, the mass or number density profiles, the velocity anisotropy profiles, as well as the optimization algorithm choices.
   * **H0 (real):** The value of the Hubble parameter evaluated at redshift z=0, measured in units of km/s/Mpc. Default is 70.0
   * **za (real):** Average redshift of the cluster's center of mass. Default is 0.
   * **Olam (real):** Value of the Omega_Lambda density parameter today. Default is 0.7
   * **Omegam (real):** Value of the Omega_m density parameter today. Default is 0.3
   * **Rlow (real):** Inner projected radius, defining the minimum projected radius from the cluster center (given in Mpc) at which galaxies should be considered in the MG-MAMPOSSt fit. Default is 0.05.
    
   * **Rup (real):** Outer projected radius, defining the maximum projected radius from the cluster center (given in Mpc) at which galaxies should be considered in the MG-MAMPOSSt fit. Default is r200g.
   *  **N(R) (string):** Number density profile model. It selects the model for the tracer's projected number density profile. In the current version three possible choices are allowed: projected NFW (**pNFW**), projected Hernquist (**pHer**) and beta-model (**beta**). Default is pNFW
    
   * **al (real):** Exponent for beta-model. The beta-model choice requires an additional input exponent which should be negative. The input parameter is ignored otherwise. Default is 0.0.
   
  * **weights (boolean integer 0/1):**  in the dataset, the user may provide a fourth column where the weights assigned to each measurement are given. If **weights=1** the code reads the additional column in the dataset. Default is 0.
   
   * **M(r) (string):** Mass profile/gravitational potential model. It selects the allowed parametrisations of the mass profile to be used in the Jeans' equation. The allowed models in GR and their corresponding **string labels** are Navarro-Frenk-White **NFW**, Hernquist **Her**, PIEMD **PIEMD**, Burkert **Bur**, Soft Isothermal model **SoftIS**, Einasto with m=5 **Eis5** and generalized NFW **gNFW**. The modified gravity frameworks are currently implemented for an effective dynamical modified NFW mass profile in linear Horndeski gravity **mNFW_LH**, modified NFW in beyond Horndeski - DHOST gravity (which exhibits Vainsthein screening) **mNFW_BH** and modified NFW in general chameleon gravity **mNFW_GC**. Default is NFW.
    
   * **Beta(r) (string):** Anisotropy profile model. It selects the  
    orbit anisotropy profile beta(r) to be used in the Jeans' analysis. The currently implemented profiles  are: constant anisotropy  **C** , Mamon&Lokas **ML** , Osipkov-Merritt **OM**, simplified Wojtak **WJ** , Tiret **T** , modified Tiret (opposite) **O**, generalized Osipkov-Merritt **gOM** and generalized Tiret **gT** .
Default is C.
    
    * **rcut (real):** Truncation radius. Value of the truncation radius needed in the pseudo-isothermal elliptical mass distribution (PIEMD), which correspond to kmp=3. It is ignored for other mass profile models. Default is 1.0.
    
    * **FASTMODE (boolean integer 0/1):** If equal to 1, the likelihood is estimated by using a grid of values (default 60 points) in the phase space (R,v_z) of the galaxies and then bispline-interpolating over the data points. Default is 0.
    
    * **OPT (integer): Optimization choice.** If required, an optimization algorithm is launched before the grid/MCMC likelihood computation to find the minimum of $-\ln \mathcal{L}$. Eventually, the resulting best fit parameters are used as guess values for the subsequent parameter space exploration. Currently, three choices are available:  BOBYQA **OPT=0**, NEWUOA **OPT=1** or POWELL **OPT=2** We point out that the POWELL algorithm may not work efficiently with gT and gOM anisotropy models, due to round-off errors in the integration routines. If **OPT=-1** the optimization is skipped. Default is -1.
    
    * **screen (integer): Screening mode** (available only for M(r)=mNFW_LH). In linear Horndeski, one can choose to adopt the $f(R)$ sub-case, **screen=0**, where A2 is fixed to 1/sqrt{6}. In this framework, there is the possibility to include a model-dependent screened $f(R)$ model with Hu&Sawicki functional form,implemented by assuming a simple analytical approximation. The transition between the screened and linear regime can be instantaneous (**screen=1**), or smoothed with an additional parameter controlling the sharpness (**screen=2**). For **screen=-1**, the general linear Horndeski with two free parameters is selected. Default is -1.
     
* **Parameter limits (lines 29-40, reals):** limits in the parameter space exploration. It works only if the option **kpro=1** in the file Options.txt. Default are always 0.0001 for the lower limit and 100 for the upper limit
   *  **r2low:** Lower limit for r200 (in Mpc). 
   *  **r2up:** Upper limit for r200 (in Mpc). 
   *  **rclow:** Lower limit for rnu (in Mpc). 
   *  **rcup:** Upper limit for rnu (in Mpc). 
   *  **rslow:** Lower limit for rs (in Mpc). 
   *  **rsup:** Upper limit for rs (in Mpc).   
   *  **blow:** Lower limit for for the first anisotropy parameter beta. For Beta(r)=ML,OM the unit is Mpc.
   *  **bup:** Upper limit for for the first anisotropy parameter beta. For Beta(r)=ML,OM the unit is Mpc.
   *  **A1low:** Lower limit for the first modified gravity parameter A1. 
   *  **A1up:** Upper limit for the first modified gravity parameter A1. 
   *  **A2low:** Lower limit for the second modified gravity parameter A2. 
   *  **A2up:** Upper limit for the second modified gravity parameter A2. 
   *  **b2low:** Lower limit for the second velocity anisotropy parameter beta2. 
   *  **b2up:** Upper limit for the second velocity anisotropy parameter beta2.

### Working Options

The file Options.txt contains various options and switches for the new features in MG-MAMPOSSt. These are mostly related to the numerical analysis and evaluation of the posterior likelihood. Notice that, the input parameters can be **binary integers** (with values 0 or 1), **integers*4** or **reals*8**. 
All the parameters must be given in a format **"label = "value"**. The "label"s are mandatory while the "value"s, if not given, are set by default. 
* **nmcmc (binary)**. Select between grid-search mode (= 0), and MCMC sampling (= 1). The default value is 1.
* **Nsample (integer)**. Number of points in the MCMC run. Default is 400000.
* **nlens (binary)**. Lensing information: If equal to 1, in the case of modified gravity mass profiles (**kmp=7,8,9**) it adds to the kinematics likelihood a probability distribution simulating additional information such as provided by a lensing mass profile reconstruction. For each set of values of the parameters, the joint (log) likelihood is then computed as

    **L(joint) =L(dyn)+L(lens),**

    where L(dyn) and L(lens) are the log-likelihoods of the MG-MAMPOSSt procedure and the simulated lensing distribution respectively. 
    For linear Horndeski and Chameleon screening, where photon propagation is not affected by the new degrees of freedom, the lensing likelihood has the form of a bivariate Gaussian distribution L(lens)=P(lens)(rs,r200), specified by central values, standard deviations and correlation (see below). In Vainsthein screening, where lensing mass profile is explicitly modified by the MG parameters, the likelihood is computed by simulating a full tangential shear profile, as explained in [Pizzuti et al., 2021](https://inspirehep.net/literature/1834246). The default value is 0.
    
 * **r200t (real)**. "True" value of the cluster's virial radius (in unit of Mpc) around which the lensing distribution is centered.
 * **rst (real)**.   "True" value of the cluster's scale radius (in unit of Mpc) around which the lensing distribution is centered.
 * **delta1 (real)**. For Vainshtein screening, it represents the intrinsic ellipticity of galaxies in the (weak) lensing simulations. For Chameleon screening, it is the relative uncertainty on the virial radius in the Gaussian distribution sigma(r_{200})/r_{200}.
 * **delta2 (real)**. For Vainshtein screening, it represents the large scale structure contribution to the errors in the (weak) lensing simulations. For Chameleon screening, it is the relative uncertainty on the scale radius in the Gaussian distribution \sigma(r_s)/r_s.
 * **delta3 (integer/real)**. For Vainshtein screening, it represents the number of galaxies per arcminute square in the (weak) lensing simulations. For Chameleon screening, it is the correlation in the Gaussian distribution.
 * **kpro (binary)**. If it is equal to 1, the parameter space exploration is made over a given interval (Xlow,Xup), where X indicates a generic free parameter. Default is 1. 
 * **Nclust (integer)**. MG-MAMPOSSt allows for efficient statistical forecast analyses of the constraints on the implemented MG models. In particular, it is possible to input Nclust realizations of phase spaces at the same time to compute directly the joint likelihood for a given set of parameters, obtained from the combination of the likelihood from each single data-set.  These data files should be located in **/data** folder and named as **datphys_"i".dat**, where "i" labels the number of the file in ascending order, starting from 2 (e.g. datphys_2.dat, datphys_3.dat, ...). The file format is the same as the main input datphys.dat. Default is 1. Note that in order to obtain meaningful results using this option, all the data files should be a realization of the same cluster (i.e. characterized by the same values of all parameters).
 * **nsingle (binary)**. If it is equal to 1, the Nclust-clusters likelihood is computed by simply multiplying by Nclust the likelihood from a single data-set. Useful to forecast a fast estimation of the limiting behaviour of the constraints when increasing the number of clusters. Default is 0.
 *  **stop (binary)**. When equal to 1, the program stops after the preliminary optimization algorithm if the relative difference between the best fit found and the guess values is larger than **epsilon=(epsilon(r_200),epsilon(r_nu),epsilon(r_s),epsilon(beta),epsilon(A_1),epsilon(A_2))**, or if the relative difference of the logarithm of the likelihood, computed at the best fit and at the guess values is larger than **delik** (see below). Default value is 0.
 *  **teps (array, real)**. Threshold values for **epsilon**. Default values are 0.1 (for all parameters).
 *  **delik (real)**. Threshold value for the relative difference of the logarithmic likelihood.  Default is 0.2.
 *  **nskip (binary)**. For the grid case, if equal to 1, the exploration starts from the free parameters guess values even if the  preliminary optimization is not skipped. This could be useful in modified gravity frameworks where the MG-MAMPOSSt likelihood presents more than one maximum, due to statistical degeneracy between model parameters. One can be interested in knowing the position of the global peak but sampling the likelihood only around a specific local maximum.
 *  **nres (binary)**. For the grid exploration without fixed bounds (**kpro=0**), it selects a larger (0) or a smaller-size (1) grid step. Note that the steps are different for each parameter and adjusted depending on the number of available tracers in the fit, unless specified (see below).
 *  **nequ (binary)**. For the grid exploration without fixed bounds (**kpro=0**), if equal to 1, it removes the re-scaling of the grid steps by the number of available tracers.
 *  **nsame (binary)**. If equal to 1, likelihoods are computed on specific values of the parameters, given from an external input file. The name of the file should be **MaxLik_input.dat**, structured as a table with seven columns (one for each parameter), following the same order as in the output likelihood **MaxLik.dat**, i.e. r200, rc, rs, beta, A1, A2, beta2. This option only works if the MCMC mode (**nmcmc=1**) is selected.
   

### Output
Output files can be found in the Output folder. The names and the location of those files can be changed by modyfing the file **gomamposst_x.inp**. The main output file is **MaxLik.dat**, 
which stores the tabulated ln(L) where L is the  likelihood/posterior, as a function of the model parameters. The other accessory files are described in the documentation.


## Test and Tutorial

### Basic smoke test
In order to check that the installation process was successfull and to produce the test example, type:
```bash
$ ./script/script_runmam.sh  
```
without changing any parameter in the input files. This way, the code should execute a 100000-points MCMC exploration of the parameter space in Vainsthein screening gravity  (**M(r)=mNFW_BH**) sampling the full kinematics+lensing likelihood by using the test data-set included in the data/ folder. The script runs in Fast Mode **FASTMODE=1** and should provide the complete output posterior in Output/MaxLik.dat within less than half an hour if the execution is performed over an average laptop. 
At the end of the run, if successufull the following text will be displayed:
```bash
 Best-fit from optimization
   r_200    =  1.406
   r_tracer =  0.330 (c_tracer=    4.26)
   r_mass   =  0.369 (c_mass=      3.81)
   Anisotropy parameter =   1.5691
   First MG parameter =  0.0828
   Second MG parameter =  0.0100
   Second Anisotropy parameter =  1.0000
 Likelihood =   5911.38999
 
   After MG-MAMPOSSt:

     build output files of binned N(R), VDP
     of best-fit N(R), VDP and of
     input N(R), VDP solutions ('true' values)

  Binned N(R) computed
  Using           13  bins for the VDP(R)
  Evaluating expected VDP for
  Max Lik solution values:
   r_200    =  1.390
   r_tracer =  0.330 (c_tracer=    4.21)
   r_mass   =  0.346 (c_mass=      4.01)
   Anisotropy parameter =   1.5457
   Second anisotropy parameter =   1.0000
   First MG parameter =  -0.0991
   Second MG parameter =  -0.0725

  sigma_r evaluated
 ```
 The plot of the marginalized distributions of the free parameters generated by the test run should correspond to what is shown by [this figure](https://github.com/Pizzuti92/MG-MAMPOSSt/blob/main/test/plot_example_test1.png). Note that the final best fit(s) can slightly chnages depending on the points sampled in the MCMC/grid.
 
### A simple tutorial of usage
In the following, we present a simple tutorial to guide the user over a step-by-step execution of the code. 
#### File name selection
First, open the gomamposst_x.inp file, which should appear as:
```bash
data/datphys_test.dat
input_pars/pars_test.txt
Output/rnvn.dat
Output/NRpar.dat
Output/NRbin.dat
Output/NRfit.dat
Output/MaxLik.dat
Output/svbin.dat
Output/svfit.dat
```
The first line is the input data-set - Be sure that the format of the data-set agrees with the documentation (the first two lines are considered as comments) - while the second line corresponds to the file storing the input parameters. All the other lines indicate the output file names. In this tutorial we will perform a test run in Chameleon Screening (**M(r)=mNFW_GC**); as such, we change the output likelihood file as
```bash
Output/MaxLik_ChameleonTest.dat
```
#### Parameter definition
Now, we define the details of the run by working on the parameters in pars_test.txt. In particular, we choose to work with the two free parameters defining the modified gravity model, which are the background value of the chameleon field **\phi_\infty**, given in unit of 1e-5 and the coupling constant **Q**. This means that the only numbers different from zero in the first block should be the fifth and the sixth:
```bash
**************  number of  steps in free parameters ********************

nr200 = 0     
               ! number of steps for r200 fit

nrc = 0        
               ! number of steps for rc fit, scale radius of N(R)
                            !   [if = 0 takes guess value]
                            !   [if = -1 forces LfM, c_nu=c]
                            !   [if = -2 fit N(R) outside MAMPOSSt]
nrs =  0       
			   ! number of steps for rs fit, scale radius of M(r)
                            !   [if = 0 takes guess value]
                            !   [if = -1 forces MfL, c=c_nu]
                            !   [if = -2 forces LCDM, c=c(M)]

nbeta = 0     
			   ! number of steps for anisotropy parameter
                            !   [if = -1 forces a_ML=r_s]
                            !   [if = -2 forces Hansen+Moore]
nA1 = 50 
			   ! number of steps in the first MG parameter 

nA2 = 30        
			   ! number of steps in the second MG parameter
			   ! If equal to -1 force the case of chameleon f(R) gravity
			   
nbeta2 = 0     ! number of steps for the second anisotropy parameter                             
************************************************************************
```
Note that, in the case of a fixed-grid exploration, this corresponds to a 51 X 31 points grid. As for the second group, we select the following guess values of the input parameters:
```bash
************* free parameters initial guess values *********************

r200g = 1.41   
			   ! mass profile r200 initial guess (Mpc)

rcg = 0.33    
			   ! N(R) scale radius initial guess (Mpc)

rsg = 0.33    
			   ! mass profile scale radius initial guess (Mpc)

betag = 1.41   
			   ! Anisotropy initial guess, beta_C, a_ML, a_OM, a_W, beta_inf 

A1g = 1.2      
			   ! first MG parameter initial guess

A2g = 0.4      
			   ! second MG parameter initial guess

beta2g = 1.0   ! Second Anisotropy parameter initial guess beta0 for 
               ! gOM and gT models
************************************************************************
```
Except for the modified gravity parameters, the others are fixed to the true values of the halo from  which the test data-set is generated. Now, let's give a look to the third group of parameters:
```bash
*************  Model options *******************************************

H0 = 70.0      
               ! Hubble constant at z=0              

za = 0.0      
               ! average redshift of the cluster (needed to evaluate Hz)                                        

Olam = 0.7     
               ! Omega lambda density parameter

Omegam = 0.3   
               ! Omega matter density parameter

Rlow = 0.05    
               ! Inner projected radius for sample selection (Mpc)

Rup = 1.41     
			   ! Outer projected radius for sample selection (Mpc)


N(R) = pNFW
               ! model for number density profile N(R)
               ! Here we list the labels for N(R) and then the corresponding profile:
               ! pNFW=projected NFW / pHer=projected Hernquist / beta=beta-model (1/2/3)
    
al = -0.00
               ! N(R) negative exponent (only if knfit=3)

M(r) = mNFW_GC	  
              ! mass profile model. 
              ! Here we list the labels for M(r) and then the corresponding profile:
              !  
              ! NFW=NFW/ Her=Hernquist/ PIEMD=PIEMD/ Bur=Burkert/ SoftIS=SoftIS/
              ! Eis5=Einasto_m=5/ mNFW_LH = mod_NFW linear Horndeski/
              ! mNFW_BH = mod_NFW beyond Horndeski/mNFW_GC= mod_NFW general chameleon/
              ! gNFW = generalized NFW
              

Beta(r) = T   
              ! Anisotropy model.
              ! Here we list the labels for beta(r) and then the corresponding profile: 
              ! C=Constant/  ML=Mamon&Lokas/ OM=Osipkov-Merritt/  WJ=simplified Wojtak/
              ! T=simplified Tiret/ O= (Opposite) modified Tiret 
              
rcut= 1.0     
              ! PIEMD model rcut in Mpc (only used if M(r)=PIEMD) 

FASTMODE= 0  
              ! Run MG-MAMPOSSt in fast mode interpolating over data points

OPT= 2        
              ! optimization algorithm: 0/1/2=bobyqa/newuao/powell
              ! -1 skip optimization
              
screen= 0   
              !     for linear f(R) kmp.eq.7, one can decide to set an instantaneous
	      !     transition between screeening and linear regime, by using the 
	      !     analytical approximation of Lombriser+12.                
              !-1/0/1/2=noscreen (general Hordenski)/noscreen f(R)/screen(instantaneous transition)
              !/screen (arctan transition)                    
			 
************************************************************************
```
The first four values represent the background cosmology (note that the test phase space has been generated at redshift 0). Then, we set the range of tracers to be considered in the MG-MAMPOSSt analysis [0.05 Mpc,1.41 Mpc] and we choose the  model **pNFW** for the number density profile of the tracers. The next parameter **-0.0** is ignored in this case.

The effective gravitational potential (mass profile) in Chameleon gravity is selected by the number **9** and the model of the velocity anisotrpy profile is chosen to be a Tiret. The following parameter 1.0 is not considered unless MG-MAMPOSSt runs with the model **PIEMD** for the mass profile. 
If we want to execute the code in fast mode, set the next paramater equal to 1, otherwise (as this is the case), type '0'. 
Finally, we choose the optimization algorithm **2** (Powell); the last parameter is ignored unless the mass profile model is **mNFW_LH**.

The last step is to choose the parameter limits for your exploration (only if **kpro=1**, see below).
```bash
********* parameter limits *********************************************

r2low= 0.4    
              !r200 lower bound

r2up= 5.0     
              !r200 upper bound

rclow= 0.05   
              !rc lower bound

rcup= 3.9     
              !rc upper bound

rslow= 0.04   
              !rs lower bound


rsup= 3.9     
              !rs upper bound

blow= 0.5    
              !beta lower bound

bup= 7.1      
              !beta upper bound

A1low = 0.006  
              !first MG parameter lower bound

A1up =   100.0
			  !first MG parameter upper bound

A2low=  0.02 
			  !second MG parameter lower bound

A2up= 100.0    
			  !second MG parameter upper bound

b2low = 0.5
			  !second beta parameter lower bound

b2up = 7.1 
			  !second beta parameter upper bound


************************************************************************
```

#### Select running options
Close the "pars_test.txt" file and open "Options.txt" to customize the execution of MG-MAMPOSSt. To choose the fixed-grid parameter space exploration, type 
```bash
nmcmc=0
```
In this case, the following parameter **Nsample** is ignored. We further exclude the additional lensing distribution by setting
```bash
nlens=0
```
To select the customized limits of the parameter space (the last group of inputs in "Pars_all_N_O_spec_DS"), set 
```bash
kpro=1
```
We leave the rest of the file unchanged. 

#### Execution
In the parent working directory, execute
```bash
$ ./script/script_runmam.sh -t  
```
or
```bash
$ sh script/script_runmam.sh -t  
```
where the additional option ```-t``` prints the execution time. If everything worked fine, at the end of the run one should obtain the following printed message:
```bash 
 Best-fit from optimization (kinematic only)
   r_200    =  1.410
   r_tracer =  0.330 (c_tracer=    4.27)
   r_mass   =  0.330 (c_mass=      4.27)
   Anisotropy parameter =   1.4100
   First MG parameter =  0.0087
   Second MG parameter =  0.0386
   Second Anisotropy parameter =  1.0000
 Likelihood =   5908.30636


  After MG-MAMPOSSt:

     build output files of binned N(R), VDP
     of best-fit N(R), VDP and of
     input N(R), VDP solutions ('true' values)

  Binned N(R) computed
  Using           13  bins for the VDP(R)
  Evaluating expected VDP for
  Max Lik solution values:
   r_200    =  1.410
   r_tracer =  0.330 (c_tracer=    4.27)
   r_mass   =  0.330 (c_mass=      4.27)
   Anisotropy parameter =   1.4100
   Second anisotropy parameter =   1.0000
   First MG parameter =   0.1197
   Second MG parameter =   0.0249

  sigma_r evaluated
******************************
time of execution:  12  seconds
******************************
No plot for the grid case
```
Note that this time of execution refers to a laptop ASUS Intel(R) Core(TM) i7-8565U CPU 1.99 GHz. The output "MaxLik_ChameleonTest.dat" file should be the same found in the "test" folder in this repository. 

In order to perform the same run in MCMC mode type ```nmcmc=1``` in "Options.txt". We select ```Nsample=100000``` points in the chain.

Running ```$ sh script/script_runmam.sh -t``` in this case should produce the [marginalized distributions](https://github.com/Pizzuti92/MG-MAMPOSSt/blob/main/test/plot_test_MCMC_Chameleon.png) in the "test" folder. The corresponding (log) likelihood is stored in "MaxLike_ChameleonTestMCMC.dat" in the same folder; the execution on a laptop with the features specified above should take roughly 10 minutes.   


## Extension and update

Since the expression of the gravitational potential is involved only in the solution of the Jeans' equation, new mass models or modified gravity/dark energy parametrizations can be introduced by modifying the routines where the potential enters, which are the functions fa(tlog) and sr2int(alr) in MAM.f. Add a new entry in the condition chain, identified by a new integer number **kmp** which should be string for **M(r)**. 






Go to [Top Page](#top)

