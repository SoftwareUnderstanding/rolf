# STaRS
Sejong Radiative Transfer through Raman and Rayleigh Scattering with atomic hydrogen

In Astronphysics, Raman spectroscopy is good tool to investigate Symbiotic Stars, Planetary Nebulae, and Active Galacitive Nuclei.

STaRS is the code for Radiative Transfer through Raman and Rayleigh Scattering with atomic hydrogen.
This code is 3D grid based Monte Carlo simulation tracing each generating photon packet.
The information of the photon packet include wavelength, position, and polarization.

The basic langauge and compiler are FORTRAN and intel FORTRAN.
I adopted parallel computing and shared memory technique for fast calculating and handling a memory.
If you have any question about the code, you send the email to "csj607@gmail.com".
Any comments for development and suggestions for collaboration are well come.

The paper for STaRS is accepted for publication in JKAS.
The title is "3D Grid-Based Monte Carlo Code for Radiative Transfer through Raman and Rayleigh Scattering
with Atomic Hydrogen --- STaRS".

In my home page, https://seokjun.weebly.com/stars.html includes the link of the paper in astro-ph and the publication list using STaRS code.

Source files

main.f90 : the main code to run STaRS

RT_grid.f90 : the module to set the scattering geometry

RT_photon.f90 : the module to generate the initial photons and describe the scattering process of photon. 

RT_obs.f90 : the module to collect the information of escaping photon.

RT_cross.f90 : the module to compute the scattering cross section and braching ratio by atomic physic.

random_mt.f90 : the module including the random generator

memory_mod.f90 : the module for the shared memory technique.

HOW TO USE THE CODE

1. Download all of file .f90 (source code) and com.sh (commands)
2. chmod +x com.sh
3. Run com.sh
3. Run STaRS.run file using the command, 'mpirun'
