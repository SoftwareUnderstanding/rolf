# License

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img
alt="Creative Commons License" style="border-width:0"
src="http://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work by
<a xmlns:cc="http://creativecommons.org/ns#"
href="https://www.lsa.umich.edu/astro/people/ci.dufujun_ci.detail"
property="cc:attributionName" rel="cc:attributionURL">Fujun Du</a> is licensed
under a <a rel="license"
href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution
4.0 International License</a>.

Recently we have published a paper (see [this
link](http://adsabs.harvard.edu/abs/2014ApJ...792....2D)) that used this code.

You are welcome to contact me at fujun.du at gmail.com or fdu at umich.edu.

# Install

Go to the ```src``` directory.  Inside it there is a ```makefile```.  You may
edit it for your own needs (but you don't have to).  Then run
```bash
    make
```
and an executable file with default name ```rac``` will be generated in the
same directory.

The ```makefile``` has a few options to compile in different environments.

## Requirements

1. Gfortran higher than 4.6.2 or Intel Fortran higher than 12.1.5 (they are
   the version I have been using for developping the code).
2. The cfitsio library.


# Run the code

There are a few input data files that are needed for the code to run.

The following files are compulsary:

    1. Configuration file.
    2. Chemical network.
    3. Initial chemical composition.
    4. Dust optical properties.

The following files are optional:

    1. Density structure.
    2. Enthalpy of formation of species.
    3. Molecular transition data.
    4. Stellar spectrum.
    5. Locations to output the intermediate steps of chemcial evolution.
    6. Species to output the intermediate steps of chemcial evolution.
    7. Species to check for grid refinement.

By default all these files are in the ```inp``` directory, though they do not
have to.  Go to this directory, and edit the file ```configure.dat``` to suit your
situation.  It has about 200 entries.  Some of them are for setting up the
physics and chemistry of the model, some are for setting up the running
environment, while others are switches telling the code whether or not it
should execute some specific tasks.  Details for editing the configure file are
included below.

After you have get the configre file ready, and have all the needed files in
place, then open a terminal and go to the directory on top of ```inp```, and
type in
```
    src/rac ./inp/configure.dat
```
to start running the code.

With the template files that are already there the code should be able to run
without any modification needed.

# Example configuration file

The configuration file is in the Fortran `namelist` format, so when editing
this file you may want to set the language type for syntax highlighting of your
editor to Fortran.

At the end of the configuration file you can write down any notes you want.
Each inline comment must be preceded by a "!", and should be separated from the
data content by at least one blank space.  They will not be read by the code.

## Thermo-chemical computation

The following configuration file can be used to run a model (but will not do
radiative transfer and ray-tracing to make images).  To make images, see below.

```fortran
! Filename: configure.dat
&grid_configure
  grid_config%rmin = 1D-1  ! Grid inner boundary
  grid_config%rmax = 200D0 ! Grid outer boundary
  grid_config%zmin = 0D0   ! Grid lower boundary
  grid_config%zmax = 200D0 ! Grid upper boundary
  grid_config%dr0  = 2D-2  ! Width of the first r step
  grid_config%columnwise = .true. ! Grid arranged in column manner
  grid_config%ncol = 200   ! Number of columns
  grid_config%use_data_file_input = .false. ! Whether to load structure from a data file
  grid_config%analytical_to_use = 'Andrews' ! Analytical type; Andrews 2009
  grid_config%interpolation_method = 'spline'
  grid_config%max_ratio_to_be_uniform = 1.5D0 ! Determines the coarseness of the grid: highest/lowest within a grid cell
  grid_config%density_scale     = 8D0 ! Roughly the density scale you are interested in; log scale of cm-3; not very important
  grid_config%density_log_range = 5D0 ! Range of density scale tou are interested in; not very important
  grid_config%max_val_considered = 1d19 ! Not used
  grid_config%min_val_considered = 1d1  ! Min density to be considered in the first few structure iterations
  grid_config%min_val_considered_use = 1d3 ! Min density actually used
  grid_config%very_small_len = 1D-6   ! Numerical accuracy of grid sizes; in AU
  grid_config%smallest_cell_size = 2D-2  ! Grid cell size will not be smaller than this; in AU
  grid_config%largest_cell_size  = 10D0  ! Largest grid cell size; in AU
  grid_config%largest_cell_size_frac  = 1D-1  ! The grid cell size should not be larger than a faction of the distance to the central star
  grid_config%small_len_frac = 5D-3     ! Similar to smallest_cell_size, but in a fractional sense
  grid_config%refine_at_r0_in_exp = .false. ! Whether to refine the grid at r0, where the disk structure makes a transition; r0 is set in later sections
/
&chemistry_configure
  chemsol_params%dt_first_step               = 1D-8  ! The first time step in year; time steps for chemical evolution is logarithmic
  chemsol_params%t_max                       = 1D6   ! Final time
  chemsol_params%ratio_tstep                 = 1.1D0 ! Ratio between width of the time steps
  chemsol_params%max_runtime_allowed         = 60.0  ! Max CPU time for each time step
  chemsol_params%RTOL                        = 1D-4  ! Relative tolerance of the result (abundances); for the ODE solver
  chemsol_params%ATOL                        = 1D-30 ! Absolute tolerance of the result
  chemsol_params%mxstep_per_interval         = 6000  ! Max number of steps for each time interval
  chemsol_params%chem_files_dir              = './inp/'  ! Directory of the chemical reaction files
  chemsol_params%filename_chemical_network   = 'rate06_withgrain_lowH2Bind_hiOBind_lowCObind.dat'
  chemsol_params%filename_initial_abundances = 'ini_abund_waterice_loMetal_CO.dat'
  chemsol_params%filename_species_enthalpy   = 'Species_enthalpy.dat'  ! Affect the chemical heating and cooling rates
  chemsol_params%H2_form_use_moeq            = .false.  ! Which kind of formula to use for the formation rate of H2; usually just set to false
  chemsol_params%flag_chem_evol_save         = .false.  ! Whether to save the chemical abundances at each time step; its content is overwritten at the next time step
  chemsol_params%evol_dust_size              = .false.  ! Not well implemented; whether to evolve the dust grain size
  chemsol_params%steps_reset_solver          = 50  ! For the solver; reset the solver after this number of steps
/
&heating_cooling_configure
  ! When use_analytical_CII_OI is true, the two files will not be used.
  heating_cooling_config%dir_transition_rates    = './transitions/'  ! Directory for the radiative files
  heating_cooling_config%use_analytical_CII_OI   = .true.  ! Whether to use analytical formula for CII and OI cooling
  heating_cooling_config%filename_CII            = ''  ! Cooling rates filename when not using analytical formula
  heating_cooling_config%filename_OI             = ''
  heating_cooling_config%IonCoolingWithLut       = .true.  ! What kind of formula to use for ion cooling
  heating_cooling_config%filename_NII            = 'N+_LUT.bin'
  heating_cooling_config%filename_SiII           = 'Si+_LUT.bin'
  heating_cooling_config%filename_FeII           = 'Fe+_LUT.bin'
  heating_cooling_config%solve_method            = 2  ! Which method to use for calculating the statistical equilibrium for radiative cooling; 1: ODE; 2: Newton
  heating_cooling_config%use_mygasgraincooling      = .true.  ! What kind of formula to use for calculating gas-grain collisional cooling; if true, use a more accurate form
  heating_cooling_config%use_chemicalheatingcooling = .true.  ! Whether to include chemical heating/cooling
  heating_cooling_config%use_Xray_heating           = .true.  ! Whether to include X-ray heating
  heating_cooling_config%heating_Xray_en            = 0.0D0  ! Ignored
  heating_cooling_config%heating_eff_chem           = 0.3D0  ! Efficiency for chemical heating/cooling
  heating_cooling_config%heating_eff_H2form         = 0.5D0  ! Efficiency for H2 formation heating
  heating_cooling_config%heating_eff_phd_H2         = 1D0  ! Efficiency for H2 photodissiation heating
  heating_cooling_config%heating_eff_phd_H2O        = 0.5D0  ! Efficiency for water photodissiation heating
  heating_cooling_config%heating_eff_phd_OH         = 0.5D0  ! Efficiency for OH photodissiation heating
  heating_cooling_config%cooling_gg_coeff           = 1D0  ! Efficiency for gas-grain collisional cooling
/
&montecarlo_configure
  mc_conf%nph                   = 4000000     ! Number of photon packets; divide total star luminosity into this number
  mc_conf%nmax_cross            = 1999999999  ! Max num of cell crossing before any absorption or scattering for a photon
  mc_conf%nmax_encounter        = 1999999999  ! Max num of absorption and scattering events for a photon
  mc_conf%ph_init_symmetric     = .true.    ! Mirror the upper disk and lower disk above and below the midplane
  mc_conf%refine_UV             = 2D-1  ! Refine for the UV photons; use smaller energy packets for them so that they have better statistics
  mc_conf%refine_LyA            = 1D-1
  mc_conf%refine_Xray           = 1D-3
  mc_conf%disallow_any_scattering  = .false.  ! If true, there will be no scattering
  mc_conf%mc_dir_in             = './inp/'  ! Input directory for some needed data
  mc_conf%mc_dir_out            = 'mc/'
  mc_conf%fname_photons         = 'escaped_photons.dat'  ! Parameters for each escaped photon are saved in this file
  mc_conf%fname_water           = 'H2O.photoxs'  ! Water absorption cross section
  mc_conf%fname_star            = 'tw_hya_spec_combined.dat'  ! Input stellar spectrum
  mc_conf%collect_photon        = .true.  ! The collected photons can be utilized to make an SED
  mc_conf%collect_lam_min       = 1D0  ! Minimum lambda; in angstrom
  mc_conf%collect_lam_max       = 1D8
  mc_conf%collect_nmu           = 4  ! Number of directions to collect photons
  mc_conf%collect_ang_mins      = 0D0   4D0    40D0  80D0  ! Ranges of angles (observing direction) to collect photons
  mc_conf%collect_ang_maxs      = 3D0   10D0   50D0  90D0
  mc_conf%nlen_lut              = 2048  ! Size of the look-up table for the dust optical parameters needed by the radiative transfer
  mc_conf%TdustMin              = 1D0  ! Lowest temperature allowed for the dust grains; in K
  mc_conf%TdustMax              = 2D3  ! Highest temperature allowed for the dust grains
  mc_conf%use_blackbody_star    = .false.  ! Whether to use blackbody spectrum for the star
  mc_conf%dist                  = 51.0D0  ! Distance of the source in pc
/
&dustmix_configure  ! To make different dust compositions
  dustmix_info%nmixture = 2  ! Number of mixtures you want to make; max: 4
  dustmix_info%lam_min  = 1D-4 ! Minimum wavelength (micron) to be considered
  dustmix_info%lam_max  = 1D4  ! Maximum ...
  !
  dustmix_info%mix(1)%id       = 1  ! Mixture 1
  dustmix_info%mix(1)%nrawdust = 3  ! Number of raw material for mixing
  dustmix_info%mix(1)%rho      = 3D0 ! Dust material density in g cm-3
  dustmix_info%mix(1)%dir          = './inp/'
  dustmix_info%mix(1)%filenames(1) = 'silicate_draine.opti'  ! Filename of raw material 1
  dustmix_info%mix(1)%filenames(2) = 'graphite_draine_pa_0.1.opti'
  dustmix_info%mix(1)%filenames(3) = 'graphite_draine_pe_0.1.opti'
  dustmix_info%mix(1)%weights(1)   = 0.8D0  ! Weight of raw material 1
  dustmix_info%mix(1)%weights(2)   = 0.04D0
  dustmix_info%mix(1)%weights(3)   = 0.16D0
  !
  dustmix_info%mix(2)%id       = 2  ! Mixture 1
  dustmix_info%mix(2)%nrawdust = 1  ! Number of raw material for mixing
  dustmix_info%mix(2)%rho      = 3D0 ! Dust material density in g cm-3
  dustmix_info%mix(2)%dir          = './inp/'
  dustmix_info%mix(2)%filenames(1) = 'silicate_draine.opti'  ! Filename of raw material 1
  dustmix_info%mix(2)%weights(1)   = 1.0D0  ! Weight of raw material 1
/
&disk_configure
  a_disk%star_mass_in_Msun         = 0.6D0  ! Mass of the central star
  a_disk%star_radius_in_Rsun       = 1D0  !  Radius of the central star
  a_disk%star_temperature          = 4000D0  ! Surface temperature of the star
  a_disk%T_Xray                    = 1D7  ! Equivalent blackbody tempearture for the X-ray spectrum of the star
  a_disk%E0_Xray                   = 0.1D0  ! Lower limit of the X-ray spectrum in keV
  a_disk%E1_Xray                   = 10D0   ! Upper limit of the X-ray spectrum in keV
  a_disk%lumi_Xray                 = 1.6D30  ! Total X-ray luminosity of the star; in erg/s
  a_disk%starpos_r                 = 0D0  ! Position of the star
  a_disk%starpos_z                 = 0D0
  !
  a_disk%use_fixed_alpha_visc      = .false.  ! If false, the alpha is not fixed
  a_disk%allow_gas_dust_en_exch    = .false.  ! Whether to let dust transfer energy back to gas; not well-implemented
  a_disk%base_alpha                = 0.01D0   ! Maximum alpha
  !
  a_disk%Tdust_iter_tandem         = .false.  ! Whether to evolve the dust temperature; not well-implemented
  !
  a_disk%waterShieldWithRadTran    = .true.  ! True means water shielding is included during the Monte Carlo radiative transfer
  !
  a_disk%andrews_gas%useNumDens    = .true. ! Gas density is the number density of hydrogen nuclei
  a_disk%andrews_gas%Md            = 2D-2   ! Total mass in Msun
  a_disk%andrews_gas%rin           = 1D-1   ! Inner edge; can be different from rmin of grid_config; in AU
  a_disk%andrews_gas%rout          = 200D0  ! Outer edge
  a_disk%andrews_gas%rc            = 80D0   ! Characteristic radius; see Andrews 2009
  a_disk%andrews_gas%hc            = 10D0   ! Scale height at boundary
  a_disk%andrews_gas%gam           = 1.5D0  ! Power index
  a_disk%andrews_gas%psi           = 1.0D0  !
  a_disk%andrews_gas%r0_in_exp     = 3.5D0  ! Inward tapering; the starting radius; in AU
  a_disk%andrews_gas%rs_in_exp     = 1D2    ! Radius scale for the tapering; in AU; very large value means a simple cutoff
  a_disk%andrews_gas%f_in_exp      = 1D-5   ! Constant scaling factor
  !
  a_disk%ndustcompo                = 3  ! Number of dust components
  !
  a_disk%dustcompo(1)%itype        = 1     ! Dust mixture type to use
  a_disk%dustcompo(1)%mrn%rmin     = 5D-3  ! Min radius; in micron
  a_disk%dustcompo(1)%mrn%rmax     = 1D3   ! Max radius
  a_disk%dustcompo(1)%mrn%n        = 3.5D0 ! Power index for the dust grain size distribution
  a_disk%dustcompo(1)%andrews%useNumDens = .false.  ! Use mass density insteady of number density
  a_disk%dustcompo(1)%andrews%Md         = 4D-4  ! Total mass in this dust component; in solar mass
  a_disk%dustcompo(1)%andrews%rin        = 1D0  ! Spatial distribution of the dust
  a_disk%dustcompo(1)%andrews%rout       = 200D0
  a_disk%dustcompo(1)%andrews%rc         = 40D0
  a_disk%dustcompo(1)%andrews%hc         = 3D0
  a_disk%dustcompo(1)%andrews%gam        = 1.5D0
  a_disk%dustcompo(1)%andrews%psi        = 1.0D0
  a_disk%dustcompo(1)%andrews%r0_in_exp  = 3.5D0
  a_disk%dustcompo(1)%andrews%rs_in_exp  = 0.5D0
  a_disk%dustcompo(1)%andrews%p_in_exp   = 3D0
  a_disk%dustcompo(1)%andrews%f_in_exp   = 1D0
  a_disk%dustcompo(1)%andrews%r0_out_exp = 45D0  ! Outer tapering; in AU
  a_disk%dustcompo(1)%andrews%rs_out_exp = 5D0
  a_disk%dustcompo(1)%andrews%f_out_exp  = 1D0
  !
  a_disk%dustcompo(2)%itype        = 1      ! Dust mixture type to use
  a_disk%dustcompo(2)%mrn%rmin     = 5D-3   ! Min radius
  a_disk%dustcompo(2)%mrn%rmax     = 1D0    ! Max radius
  a_disk%dustcompo(2)%mrn%n        = 3.5D0  ! Power index
  a_disk%dustcompo(2)%andrews%useNumDens = .false. ! Use mass density insteady of number density
  a_disk%dustcompo(2)%andrews%Md         = 2D-5
  a_disk%dustcompo(2)%andrews%rin        = 1D-1
  a_disk%dustcompo(2)%andrews%rout       = 200D0
  a_disk%dustcompo(2)%andrews%rc         = 80D0
  a_disk%dustcompo(2)%andrews%hc         = 20D0
  a_disk%dustcompo(2)%andrews%gam        = 0.5D0
  a_disk%dustcompo(2)%andrews%psi        = 1.0D0
  a_disk%dustcompo(2)%andrews%r0_in_exp  = 3.5D0
  a_disk%dustcompo(2)%andrews%rs_in_exp  = 0.5D0
  a_disk%dustcompo(2)%andrews%p_in_exp   = 3D0
  a_disk%dustcompo(2)%andrews%f_in_exp   = 1D0
  !
  a_disk%dustcompo(3)%itype        = 2      ! Dust mixture type to use
  a_disk%dustcompo(3)%mrn%rmin     = 0.9D0  ! Min radius
  a_disk%dustcompo(3)%mrn%rmax     = 2D0    ! Max radius
  a_disk%dustcompo(3)%mrn%n        = 3.5D0  ! Power index
  a_disk%dustcompo(3)%andrews%useNumDens = .false. ! Use mass density insteady of number density
  a_disk%dustcompo(3)%andrews%Md         = 1D-9
  a_disk%dustcompo(3)%andrews%rin        = 1D-1
  a_disk%dustcompo(3)%andrews%rout       = 3.5D0
  a_disk%dustcompo(3)%andrews%rc         = 80D0
  a_disk%dustcompo(3)%andrews%hc         = 10D0
  a_disk%dustcompo(3)%andrews%gam        = 1.0D0
  a_disk%dustcompo(3)%andrews%psi        = 1.0D0
  a_disk%dustcompo(3)%andrews%r0_in_exp  = 0.4D0
  a_disk%dustcompo(3)%andrews%rs_in_exp  = 0.1D0
  a_disk%dustcompo(3)%andrews%p_in_exp   = 2D0
  a_disk%dustcompo(3)%andrews%f_in_exp   = 1D0
/
&raytracing_configure  ! Not needed when you don't do ray-tracing calculations (to generate spectra and images)
  raytracing_conf%dirname_mol_data       = './transitions/'  ! Directory for the radiative transfer parameters
  raytracing_conf%fname_mol_data         = '12C16O_H2.dat'  ! Filename of the transition rates
  raytracing_conf%line_database          = 'lamda'  ! Data format
  raytracing_conf%mole_name_disp         = 'C$^{18}$O'  ! Actually calculating for C18O, so use this name
  raytracing_conf%maxx = 2.0000e+02  ! For making images; box size in AU; (-maxx, maxx)
  raytracing_conf%maxy = 2.0000e+02
  raytracing_conf%nx                     = 401  ! Number of pixels in each direction
  raytracing_conf%ny                     = 401
  raytracing_conf%nfreq_window           = 3  ! Only care about lines within these frequency windows
  raytracing_conf%freq_mins              = 0.9D11  3.4D11  6.90D11  ! Ranges of the frequency windows; in Hz
  raytracing_conf%freq_maxs              = 3.0D11  3.6D11  6.92D11
  raytracing_conf%nf                     = 200  ! Number of spectral channels
  raytracing_conf%VeloKepler             = 3D4  ! A rough estimation of the highest Keplerian velocity in m/s; for calculating the line velocity range
  raytracing_conf%E_min                  = 0D0  ! Range of energy level of transitions we care about; in K
  raytracing_conf%E_max                  = 3D3
  raytracing_conf%min_flux               = 0D-2  ! Minimum line flux to be saved; in Jy
  raytracing_conf%save_spectrum_only     = .false.  ! If true, the image will not be saved; only the total spectrum is saved
  raytracing_conf%nlam_window            = 6  ! Only for continuum; make continuum spectra in these wavelength ranges; in micron
  raytracing_conf%lam_mins               = 1D-4  1D-3  0.1  1.0   10.0   100.0
  raytracing_conf%lam_maxs               = 1D-3  1D-2  1.0  10.0  100.0  1000.0
  raytracing_conf%nlam                   = 100  ! Lambda channels
  raytracing_conf%abundance_factor       = 2.0D-3  ! For modeling isotopologues when the transition rates of itself is not available
  raytracing_conf%useLTE                 = .true.  ! Whether use LTE
  raytracing_conf%nth                    = 1  ! Number of viewing angles (theta)
  raytracing_conf%view_thetas = 7.0000e+00  ! Viewing angles in degree
  raytracing_conf%dist = 5.1000e+01  ! Distance of the source in pc
  raytracing_conf%VeloTurb               = 200D0  ! Turbulent line broadening; in m/s
  raytracing_conf%VeloWidth              = 30D0  ! Additional velocity width added to the spectra velocity range; in m/s
/
&cell_configure
  cell_params_ini%omega_albedo              = 0.5D0  ! Dust albedo, only for chemistry
  cell_params_ini%UV_G0_factor_background   = 1D0    ! ISM UV
  cell_params_ini%zeta_cosmicray_H2         = 1.36D-17  ! Cosmic ray intensity
  cell_params_ini%PAH_abundance             = 1.6D-9  ! PAH abundance; for heating/cooling
  cell_params_ini%GrainMaterialDensity_CGS  = 2D0  ! Density of dust material; not used
  cell_params_ini%MeanMolWeight             = 1.4D0  ! Mean molecular weight of the gas
  cell_params_ini%alpha_viscosity           = 0.01D0  ! Alpha; not used
/
&analyse_configure
  ! Do chemical analysis for some species at some locations
  a_disk_ana_params%do_analyse                      = .true.
  a_disk_ana_params%analyse_points_inp_dir          = './inp/'
  a_disk_ana_params%file_list_analyse_points        = 'points_to_analyse.dat'
  a_disk_ana_params%file_list_analyse_species       = 'Species_to_analyse.dat'
  a_disk_ana_params%file_analyse_res_ele            = 'elemental_reservoir.dat'  ! Which species contain the most of one element
  a_disk_ana_params%file_analyse_res_contri         = 'contributions.dat'  ! Which reactions are the most important for the creation and destruction of each species
/
&iteration_configure
  a_disk_iter_params%n_iter                         = 1  ! Number of iterations
  a_disk_iter_params%nlocal_iter                    = 4  ! For numerical stability
  a_disk_iter_params%do_vertical_struct             = .true.  ! Whether do the vertical structure calculation
  a_disk_iter_params%do_vertical_with_Tdust         = .true.  ! Whether to use dust temperature for vertical structure calculation
  a_disk_iter_params%calc_Av_toStar_from_Ncol       = .false.  ! Whether to calculate Av by the column density from each location towards the star
  a_disk_iter_params%calc_zetaXray_from_Ncol        = .false.  ! Whether to calculate X-ray ionization rate based on the column density from each location towards the star
  !
  a_disk_iter_params%rescale_ngas_2_rhodust         = .true.  ! If true, the gas density is a simple scaling of the dust density
  a_disk_iter_params%dust2gas_mass_ratio_deflt      = 1.0D-2  ! Dust-to-gas mass ratio used for calculating the gas density from the dust density using simple scaling
  a_disk_iter_params%max_num_of_cells               = 10000  ! Maximum number of grid cells
  !
  a_disk_iter_params%deplete_oxygen_carbon          = .true.  ! If true, will deplete oxygen and carbon
  a_disk_iter_params%deplete_oxygen_carbon_method   = 'vscale'  ! How to deplete oxygen and carbon
  a_disk_iter_params%deplete_oxygen_method          = 'tanh'
  a_disk_iter_params%deplete_carbon_method          = 'tanh'
  a_disk_iter_params%tanh_OC_enhance_max            = 1D3  ! Oxygen and carbon might also be enhanced, by at most this factor
  a_disk_iter_params%gval_O                         = 1D-2  ! Smallest oxygen depletion factor
  a_disk_iter_params%tanh_r_O                       = 15D0  ! Transitional radius of the depletion; in AU
  a_disk_iter_params%tanh_scale_O                   = 3D0  ! Radius scale of the gradual transitional region
  a_disk_iter_params%tanh_minval_O                  = 0.6D0  ! Minimum scale height of oxygen relative to the gas scale height
  a_disk_iter_params%tanh_maxval_O                  = 0.9D0  ! Maximum ...
  a_disk_iter_params%gval_C                         = 1D-2
  a_disk_iter_params%tanh_r_C                       = 60D0
  a_disk_iter_params%tanh_scale_C                   = 5D0
  a_disk_iter_params%tanh_minval_C                  = 0.2D0
  a_disk_iter_params%tanh_maxval_C                  = 0.7D0
  !
  a_disk_iter_params%do_vertical_every              = 1  ! How often do we recalculate the vertical structure (if the whole calculation iterates many times)
  a_disk_iter_params%nVertIterTdust                 = 16  ! Maximum number of iteration for dust Monte Carlo radiative transfer
  a_disk_iter_params%rtol_abun                      = 0.2D0  ! Tolerance for checking the convergency of a cell; only when doing chemical iterations
  a_disk_iter_params%atol_abun                      = 1D-12
  a_disk_iter_params%converged_cell_percentage_stop = 0.95  ! Stop if this fraction of cells have converged.
  a_disk_iter_params%n_gas_thrsh_noTEvol            = 1D15  ! Grid cells with density higher than this value will simply take the dust temperature as the gas temperature; in cm-3
  a_disk_iter_params%filename_list_check_refine     = 'species_check_refine.dat'  ! Do grid cell refinement based on species in this file.
  a_disk_iter_params%threshold_ratio_refine         = 10D0  ! Do refinement if the gradient (ratio) is so large
  a_disk_iter_params%nMax_refine                    = 100   ! Max times of refinments
  a_disk_iter_params%redo_montecarlo                = .true.  ! Redo Monte Carlo after each full chemcial run, which will recalculate the dust temperatures, UV fields, ...
  a_disk_iter_params%flag_save_rates                = .false.  ! Whether save the calculated rates.
  !
  a_disk_iter_params%do_continuum_transfer          = .false.  ! Related to raytracing_configure
  a_disk_iter_params%do_line_transfer               = .false.
  !
  a_disk_iter_params%backup_src                     = .true. ! If true, it will backup files using the command backup_src_cmd
  a_disk_iter_params%backup_src_cmd                 = 'find src/*.f90 src/*.f src/makefile inp/*dat inp/*opti | cpio -pdm '
  !
  a_disk_iter_params%dump_common_dir                = './tmp_data_dump/'  ! Common folder for backing up binary data
  a_disk_iter_params%dump_sub_dir_out               = 'current_model_name/'  ! Subfolder in the common folder for the current model
  a_disk_iter_params%iter_files_dir                 = 'common_results_folder/current_model_name/'  ! For saving all the model data (except for the binary files)
/
! Write any notes below
```

## Radiative transfer

After the thermo-chemical calculations are done, **replace** the
`iteration_configure` section of the previous configuration file with the
following, then rerun the model to do radiative transfer and make images.  It
will load the data (saved in binary format) generated by the previous step.
Here we are doing radiative transfer for C18O, and assume the
`raytracing_configure` section of the previous configuration file is already
correctly set for this, otherwise that section needs to be modified
accordingly.

```fortran
&iteration_configure
  a_disk_iter_params%n_iter                         = 1  ! Number of iterations
  a_disk_iter_params%nlocal_iter                    = 4  ! For numerical stability
  a_disk_iter_params%do_vertical_struct             = .false. ! Whether do the vertical structure calculation
  a_disk_iter_params%do_vertical_with_Tdust         = .true.  ! Whether to use dust temperature for vertical structure calculation
  a_disk_iter_params%calc_Av_toStar_from_Ncol       = .false.  ! Whether to calculate Av by the column density from each location towards the star
  a_disk_iter_params%calc_zetaXray_from_Ncol        = .false.  ! Whether to calculate X-ray ionization rate based on the column density from each location towards the star
  !
  a_disk_iter_params%rescale_ngas_2_rhodust         = .true.  ! If true, the gas density is a simple scaling of the dust density
  a_disk_iter_params%dust2gas_mass_ratio_deflt      = 1.0D-2  ! Dust-to-gas mass ratio used for calculating the gas density from the dust density using simple scaling
  a_disk_iter_params%max_num_of_cells               = 10000  ! Maximum number of grid cells
  !
  a_disk_iter_params%deplete_oxygen_carbon          = .true.  ! If true, will deplete oxygen and carbon
  a_disk_iter_params%deplete_oxygen_carbon_method   = 'vscale'  ! How to deplete oxygen and carbon
  a_disk_iter_params%deplete_oxygen_method          = 'tanh'
  a_disk_iter_params%deplete_carbon_method          = 'tanh'
  a_disk_iter_params%tanh_OC_enhance_max            = 1D3  ! Oxygen and carbon might also be enhanced, by at most this factor
  a_disk_iter_params%gval_O                         = 1D-2  ! Smallest oxygen depletion factor
  a_disk_iter_params%tanh_r_O                       = 15D0  ! Transitional radius of the depletion; in AU
  a_disk_iter_params%tanh_scale_O                   = 3D0  ! Radius scale of the gradual transitional region
  a_disk_iter_params%tanh_minval_O                  = 0.6D0  ! Minimum scale height of oxygen relative to the gas scale height
  a_disk_iter_params%tanh_maxval_O                  = 0.9D0  ! Maximum ...
  a_disk_iter_params%gval_C                         = 1D-2
  a_disk_iter_params%tanh_r_C                       = 60D0
  a_disk_iter_params%tanh_scale_C                   = 5D0
  a_disk_iter_params%tanh_minval_C                  = 0.2D0
  a_disk_iter_params%tanh_maxval_C                  = 0.7D0
  !
  a_disk_iter_params%do_vertical_every              = 1  ! How often do we recalculate the vertical structure (if the whole calculation iterates many times)
  a_disk_iter_params%nVertIterTdust                 = 16  ! Maximum number of iteration for dust Monte Carlo radiative transfer
  a_disk_iter_params%rtol_abun                      = 0.2D0  ! Tolerance for checking the convergency of a cell; only when doing chemical iterations
  a_disk_iter_params%atol_abun                      = 1D-12
  a_disk_iter_params%converged_cell_percentage_stop = 0.95  ! Stop if this fraction of cells have converged.
  a_disk_iter_params%n_gas_thrsh_noTEvol            = 1D15  ! Grid cells with density higher than this value will simply take the dust temperature as the gas temperature; in cm-3
  a_disk_iter_params%filename_list_check_refine     = 'species_check_refine.dat'  ! Do grid cell refinement based on species in this file.
  a_disk_iter_params%threshold_ratio_refine         = 10D0  ! Do refinement if the gradient (ratio) is so large
  a_disk_iter_params%nMax_refine                    = -1    ! Max times of refinments
  a_disk_iter_params%redo_montecarlo                = .true.  ! Redo Monte Carlo after each full chemcial run, which will recalculate the dust temperatures, UV fields, ...
  a_disk_iter_params%flag_save_rates                = .false.  ! Whether save the calculated rates.
  !
  a_disk_iter_params%do_continuum_transfer          = .false.  ! Related to raytracing_configure
  a_disk_iter_params%do_line_transfer               = .true.
  !
  a_disk_iter_params%backup_src                     = .false. ! If true, it will backup files using the command backup_src_cmd
  a_disk_iter_params%backup_src_cmd                 = 'find src/*.f90 src/*.f src/makefile inp/*dat inp/*opti | cpio -pdm '
  !
  a_disk_iter_params%dump_filename_chemical         = 'chemical_data_iter_0001.bin'  ! Chemical abundances obtained from the model calculation
  a_disk_iter_params%dump_filename_physical_aux     = 'physical_data_aux_iter_0001.bin'  ! Physical parameters obtained from the model calculation
  a_disk_iter_params%dump_filename_grid             = 'grid_data_iter_0001.bin'  ! Grid parameters obtained from the model calculation
  a_disk_iter_params%dump_filename_physical         = 'physical_data_iter_0001.bin'
  a_disk_iter_params%dump_filename_optical          = 'optical_data_iter_0001.bin'
  a_disk_iter_params%use_backup_chemical_data       = .true.
  a_disk_iter_params%use_backup_grid_data           = .true.
  a_disk_iter_params%use_backup_optical_data        = .true.
  a_disk_iter_params%dump_sub_dir_in                = 'current_model_name/'
  !
  a_disk_iter_params%dump_common_dir                = './tmp_data_dump/'  ! Common folder for backing up binary data
  a_disk_iter_params%dump_sub_dir_out               = 'current_model_name/'  ! Subfolder in the common folder for the current model
  a_disk_iter_params%iter_files_dir                 = 'common_results_folder/current_model_name/C18O/'  ! If you are modeling C18O; all the images will be saved under C18O/images/
/
```

```python
def load_data_as_dic(filepath, comments='!', returnOriginalKeys=False):
    import numpy as np
    
    data = np.loadtxt(filepath, comments=comments)

    ftmp = open(filepath, 'r')
    str_comment = ftmp.readline()[1:].split()
    ftmp.close()

    dic = {}
    for i in xrange(len(str_comment)):
        dic.update({str_comment[i]: data[:, i]})

    del data

    if returnOriginalKeys:
        return str_comment, dic
    else:
        return dic
```
