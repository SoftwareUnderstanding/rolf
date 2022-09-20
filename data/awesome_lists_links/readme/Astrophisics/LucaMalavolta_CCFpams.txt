# CCFpams: Atmospheric Stellar Parameters from Cross-Correlation Functions

**`CCFpams` v1.0 by Luca Malavolta - 2017**    

CCFpams is a novel approach that allows the measurement of stellar temperature,
metallicity and gravity within a few seconds and in a completely automated fashion. Rather than performing comparisons with spectral libraries, our technique is based on the determination of several cross-correlation functions (CCFs) obtained by including spectral features with different sensitivity to the photospheric parameters. We use literature stellar parameters of high signal-to-noise (SNR), high-resolution HARPS spectra of FGK Main Sequence stars to calibrate the stellar parameters as a function of CCF areas.  For FGK
stars we achieve a precision of 50K in temperature, 0.09 dex in gravity and 0.035 dex in metallicity at SNR=50 while the precision for observation with SNR>100 and the overall accuracy are constrained by the literature values used to calibrate the CCFs.

The paper describing the technique has been published by MNRAS ([Malavolta et al. 2017](http://adsabs.harvard.edu/abs/2017MNRAS.469.3965M)). In this repository the code to extract the parameters from HARPS and HARPS-N is provided.

## CCFpams user guide

1. [Install and Compile](#install-and-compile)
2. [Prepare the observations](#prepare-the-observations)
3. [Run the code](#run-the-code)
4. [Analyze the output](#analyze-the-output)

### Install and Compile

`CCFpams` is avaialble on [GitHub](https://github.com/LucaMalavolta/CCFpams/ "CCFpams repository")

Download the .zip files or clone the repository. `harps_input2pams.f90` and `harpn_input2pams.f90` are the two FORTRAN90 codes to compute the stellar parameters from HARPS and HARPS-N data respectively.

 The subroutines required by the two programs are inside the `routines_f90` folder. Calibration files required by the two programs are stored in the `mask_calib` folder.

 Before compiling, you have to declare the path of the code in the code itself. In this way the code will be able to find automatically all the required calibration files. Additionally you can specify the directory of the HARPS/HARPS-N archive of the _DRS reduced_ files.

To do so, open `harps_input2pams_v3.f90`/`harpn_input2pams_v3.f90` with a text editor and change `code_path` and (optionally) `archive_harpn` with the full path of your folders.

```fortran
  character (len=*), parameter :: &
     code_path = '/home/malavolta/CODE/CCFpams/', &
     archive_harpn = '/home/malavolta/data/HARPN/data/'
```

To compile the program, just execute in a shell:

```sh
 $ ./COMPILE
```
The script will create two executable files, `harps_input2pams.e` and `harpn_input2pams.e`

### Prepare the observations

The program requires a file where all the observations to be analyzed are listed.
The file list must include:
- the date when the observations have been gathered (the value at the beginning of the night is taken as reference),
- the filename of the raw data, without extension,
- the mask that have been used by the HARPS/HARPS-N pipeline (shortly _DRS_) to reduce the data and extract the CCF.
Its structure must follow this example:
```text
2014-07-08   HARPN.2014-07-09T01-15-47.342   G2
2014-07-09   HARPN.2014-07-10T01-43-07.290   G2
2014-07-10   HARPN.2014-07-11T00-07-59.866   G2
...
```
The size of the space between the columns is not relevant.

The DRS automatically stores the reduced data of calibrations and observations in a directory named as the night of the observations. So the easiest way to use this program is just to copy somewhere the entire folder.  The code will use the information in the list file to locate the 2-dimensional extracted spectrum (```e2ds``` file), the the CCF extracted by the DRS (to correct for the barycentric earth RV and remove the RV of the star), and the blaze function of the spectrograph. All these files must be located within the same directory.

Alternatively, all the files from different nights can be located in the same folder, by specifying the appropriate flag when running the code.

### Run the code

A short description of the options required by the code is given when the program is execute with no option at all:

```text
$ ./harpn_input2pams.e

Input 1: object list file
Input 2: output rad (usually the object name)
Input 3: (optional): archive directory
Input 4: (optional): 0 = do not use date as prefix
Input 4  to be used if all the files are in the same directory (as Yabi reprocessed files are)
Input 5: (optional): verbosity on
```

Here a detailed description:
1. [object list file]: the file list as described in the previous section. If the code is not executed in the directory where the file is located, the full path must be included.
2. [output prefix]: this prefix will be used for the name of the output files.
3. [archive directory]: the directory where either the night directory or the whole ensemple of files are sotred. If not provided, the ```archive_harps```/```archive_harpn``` variable hard-coded in the program will be used.
4. [use the night directory?]: optional, default is ```1=yes```. If ```0=no``` then the prigram will look for the observation files directly in the archive directory, without crawling in the night subdirectories.
5. [verbosity]: optional, if provided a huge amount of intermediate files will be produced. It is meant for debugging work, the program is already logorrheic with this option turned off.

To use any optional argument, all the previous arguments must be specified, even if they are set to the default value.

Here an example:
```text
$ /Users/malavolta/Astro/CODE/CCFpams/harpn_input2pams.e Kepler19.list Kepler19 /Users/malavolta/Astro/HARPS-N/archive/ 1
```

The programm will start producing a terminal output listing the calibration files that have been read:
```text
opening file:/Users/malavolta/Astro/CODE/CCFpams/mask_calib/TGdirect_666_direct_calib_cheb.dat
Sunsky_NewMix_neu_hgh
Sunsky_NewMix_neu_med
Sunsky_NewMix_neu_low
opening file:/Users/malavolta/Astro/CODE/CCFpams/mask_calib/LOGG_ewfind_33222_ewfind_logg4_calib_cheb.dat
Sunsky_NewMix_ion_all
OPENING linelist file: Sunsky_NewMix_neu_hgh_linelist.dat
OPENING linelist file: Sunsky_NewMix_neu_med_linelist.dat
OPENING linelist file: Sunsky_NewMix_neu_low_linelist.dat
OPENING linelist file: Sunsky_NewMix_ion_all_linelist.dat
```

Then for each observation in the list file a short log is given:
```text
Doing spectrum HARPN.2014-07-10T01-43-07.290
OUTNAME_sum ../mask_ccf/Kepler19/HARPN.2014-07-10T01-43-07.290_sccf.fits.gz
  2.5299929973006412       0.28598160253350391        1.8136243945829278
  2.4010608651684309       0.32357900378018190        1.9474819314337917
  2.4228101390858408       0.37525332690862845        2.2789451252072768
  2.2943462316515113       0.29226754946282196        1.6808520522550257
AREA_out    1.8136243945829278        1.9474819314337917        2.2789451252072768        1.6808520522550257
Parameters:    5551.2563301093069      -0.10045953834263877        4.5140399932861328
Processing of spectra HARPN.2014-07-10T01-43-07.290  completed
```
If you see something different from this, something has went wrong.

### Analyze the output

The code automatically creates a folder called  ```output_dir``` in the same directory where the program has been executed. When the verbosity option is off, two files with appended ```_outcal.dat``` and ```_outcal_extended.dat``` to the prefix given as input will be created. If the files already exists, the program will crash (by design, to avoid file overwriting).

The ```_outcal.dat``` is structured as in the following:
```
1     1    2456847.566087  5558.31    69.59    -0.09011     0.05322     4.52439     0.12797     1.82857     1.96015     2.28516     1.68259     31.70     1.0154764   HARPN.2014-07-09T01-15-47.342
2     2    2456848.584943  5551.26    67.18    -0.10046     0.05167     4.51404     0.12371     1.81362     1.94748     2.27895     1.68085     34.60     1.0307872   HARPN.2014-07-10T01-43-07.290
3     3    2456849.518890  5584.66    68.47    -0.10067     0.05250     4.53778     0.12599     1.81210     1.92388     2.26875     1.67452     33.00     1.0237369   HARPN.2014-07-11T00-07-59.866
4     4    2456850.548335  5565.35    80.35    -0.14536     0.06018     4.55267     0.14697     1.74871     1.85414     2.24616     1.59305     22.50     1.0130088   HARPN.2014-07-12T00-49-51.416
....
0     0          9.795921  5540.44    44.31    -0.11157     0.03770     4.49000     0.08394     1.80268     1.93808     2.28371     1.68039    117.55     0.0000000   COADDED
```

1. Index of the observation, as encountered in the list file.
2. Index of the processed observation, it will differ from the previous index if something went want with one or more observations.
3. BJD of the observation.
4. Effective temperature of the star, as determined by the program.
5. Error on the effective temperature.
6. Metallicity of the star, as determined by the program.
7. Error on the metallicity.
8. Gravity of the star, as determined by the program.
9. Error on gravity.
10. to 13. : CCF areas for each of the mask used in the analysis.
14. SNR of the observation .
15. Airmass of the observation.
16. Name prefix of the observation

TO BE COMPLETED
