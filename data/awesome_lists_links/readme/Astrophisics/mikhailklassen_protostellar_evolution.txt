# Protostellar Evolution

A code for simulating the evolution of stellar properties (stellar radius, luminosity) from the bound core stage through to the core hydrogen ignition as a zero-age main-sequence (ZAMS) star and beyond.

This code is written in Fortran 90 for my M.Sc. thesis work in astrophysics at McMaster University. The [entire thesis](http://digitalcommons.mcmaster.ca/opendissertations/6290/) can be downloaded from the [DigitalCommons](http://digitalcommons.mcmaster.ca/opendissertations/6290/) open access dissertations and theses archive at McMaster University.

This code was implemented as a module in [FLASH](http://flash.uchicago.edu) astrophysical fluid dynamics code developed by the University of Chicago. Simulations conducted with this code studying protostellar evolution and stellar radiation feedback formed the basis of the following two papers:

- [**H II Region Variability and Pre-main-sequence Evolution**](http://adsabs.harvard.edu/abs/2012ApJ...758..137K). Klassen, Peters, & Pudritz. 2012. *The Astrophysical Journal*, **758**, 2, 137.
- [**Simulating protostellar evolution and radiative feedback in the cluster environment**](http://adsabs.harvard.edu/abs/2012MNRAS.421.2861K). Klassen, Pudritz, & Peters. 2012. *Monthly Notices of the Royal Astronomical Society*. **421**, 4, 2861

## Use

To compile the code, adjustments may need to be made to the `Makefile` to set the Fortran compiler and compiler flags. The main program code is in `Driver.F90`, which calls `EvolveProtostellar`. In `Driver.F90`, the timestep `dt` and accretion rate `mdot` are set. These can be changed manually to run the code with different accretion rates or with a different timestep size.

When run, a file `protostellar_evolution.txt` is created and the stellar properties are written to this file as the protostar evolves with the simulation. All units are in the [CGS](http://en.wikipedia.org/wiki/Centimetre%E2%80%93gram%E2%80%93second_system_of_units) system, as is the standard in astrophysics.

## Precomputed Tables

The protostellar evolution code relies on two precomputed tables: `modeldata_table.dat` and `beta_table.dat`. These have been precomputed and are present in the root directory. In order to recreate these tables, the code in the subdirectory 'table\_generators' needs to be compiled and run. There is a separate `Makefile` for the code in this subdirectory. Once compiled, it must only be run to recreate the precomputed tables. This process can take some time however.
