# PPInteractions
Installation instructions:
download and install gfortran: https://gcc.gnu.org/wiki/GFortranBinaries

Running Instructions : 
Before you run the code, go into the section "PARAMETERS TO FIX", and:
- choose the parameters of the primary proton spectrum:
    *alpha1, E_cut (in TeV), Norm (in 1/(TeVcm^3)), alpha2,	E_0 (in TeV).
  These parameters describe the proton spectrum, as:                 
    * J_Pr=Norm*(E_Pr(i_p)/E_0)**(-alpha1)*exp(-(E_Pr(i_p)/E_cut)**(alpha2)) (following the notations of Kelner et al (2006), Eq     (72).) 	 
- choose the maximal and minimal energy of the primary protons in TeV: E_Pr_min, E_Pr_max;  
- choose the density of the target proton in cm^-3;
- choose the maximal and minimal energy of the secondary particle in TeV: Emin, Emax 

To run the code type
- gfortran -g -fimplicit-none pp_interaction_transition_public.f
- ./a.out 

This code allows you to obtain secondary particle spectra for any value of alpha1, over the entire energy range delimited by Emin and Emax.	The code's running time may be a bit long (~30 minutes) because all the subroutines are called twice: once to compute the high energy part of the particle spectra and a second time, to adjust the low energy part of the spectra to the high energy one.

The produced secondary particle spectra are given in units 1/(TeVcm^3s). You can choose which secondary particle spectra you want to record into a file in the section "writ". As an example, for the moment this code creates the files 'proton_spectrum.dat', which corresponds to the spectra of the primary protons; the file 'photon_spectrum.dat', which corresponds to the spectrum of the photons produced in pion0 decay; and the file 'nu_allflavor_spectrum.dat', where the spectrum of the neutrinos of all flavors is recorded.  

Please do not hesitate to contact me if you have any comments. My e-mail address is: Celine.Tchernin@unige.ch. 
