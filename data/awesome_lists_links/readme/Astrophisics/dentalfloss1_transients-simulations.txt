# transient-simulations

## Requirements

Python 3.6 or greater with the following libraries:
* numpy
* scipy
* matplotlib
* tqdm 
* bitarray
* astropy

Should be platform independent

## Installing

1. Make a virtual environment: python3 -m venv mytransientsvenv
2. Activate the virtual environment: (in linux it is) ```source mytransientsvenv/bin/activate``` For other platforms use the correct activate file. 
3. Install the above dependencies using ```python3 -m pip install numpy``` etc

Optionally, you can try installing the above packages individually and create a virtual environment, but troubleshooting will be more challenging.


## Running the Simulation

1. Edit the config.ini file to your liking
2. Specify a observation file (or fill out the observation parameters in the config.ini file)
3. Run simulation using:
``` python3 simulate.py --observations myobsfile.txt```
4. Program runs and dumps out a bunch of plots and numpy arrays. Move them to a folder when it's completed so that they don't get overwritten by additional runs.


## Adding lightcurves

Lightcurves are imported dynamically by calling whatever is in the lightcurve type field in the config.ini file. 
For example, if there is a lightcurve class file called "example.py"  then all one has to do is specify "example"
in the lightcurvetype variable. 

The structure of these files should be easy to copy by taking the existing lightcurves as examples. The procedure 
generally should be as follows:

1. Specify whether the lightcurve has definite edges
2. Define the earliest and latest critical times that the lightcurve can be simulated for. 
*Note: Lightcurves with a definite beginning must have a critical time at
the beginning of the lightcurve*
3. Specify function for the integrated flux
4. Specify functions for the lines of the expected probability of 1 

