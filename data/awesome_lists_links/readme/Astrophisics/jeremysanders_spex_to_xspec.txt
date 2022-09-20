# spex_to_xspec

spex_to_xspec is a Python program to generate an APEC-format table model (http://www.atomdb.org/) from the SPEX spectral modelling and fitting code  (https://www.sron.nl/astrophysics-spex). This allows the user to load a SPEX-generated plasma model into Xspec (https://heasarc.gsfc.nasa.gov/lheasoft/xanadu/xspec) to use the same environment for both APEC and SPEX spectral fitting.

## Getting Started

Download the spex_to_xspec.py script (either using `git clone` or download the ZIP file in github). You will need to setup the SPEX environment (set the `SPEX90` environment variable and source its initialisation scripts).

There are some adjustable parameters you can modify at the top of the script.  These include the default filename root part (to be used in Xspec when loading the APEC table), the list of temperatures to tabulate the SPEX model for and the continuum energy binning.

When you are ready, run the script using python.

By default the temperature grid is the same as the APEC model (in apec-3.0.9+; 201 bins in log space between 0.00086 and 86 keV). I suggest doing an initial run with a smaller grid to make sure everything is ok, as a full run can take considerable time (several hours, depending on CPU) and several tens of GB disk space. You can compress the output files with gzip and Xspec is able to read them.

By default weak lines are dumped into a "pseudo-continuum" to save space in the output files and improve the speed of the model in Xspec. You can control the line strength where this happens by modifying the variable `minepsilon`. This can be set to `None` to disable putting these lines into a pseduo-continuum.

Note that the program dumps out a lot of data into temporary files in the directory workdir (called `tmp_*`), which are later interpreted. If you stop the program mid way through, you will need to clean up these files.

To use the table in xspec use
```
xset APECROOT spex
```
Where `spex` is the root specified at the top of the script.

### Prerequisites

 1. Python 3.3+
 2. Astropy (http://www.astropy.org/)
 3. SPEX 3.06.01+ (https://www.sron.nl/astrophysics-spex). 3.06.01 has been tested.

## How it works

The script generates a line list and a continuum for each of the temperature grid points in ASCII format from SPEX. It then interprets these and converts them to FITS format. Generating the line list file is fairly simple, but to calculate the continuum for each element, the continuaa are dumped with very high metallicity so that the hydrogen and helium continuaa can be subtracted with good numerical precision.

By default the script switches to the latest SPEX plasma model (`var calc new`).


## Future improvements

 1. Clean up temporary files on incomplete run
 2. Read output files as they are generated to avoid large temporary space usage

## Author

* **Jeremy Sanders**  - [jeremysanders](https://github.com/jeremysanders)

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details
