# Radio Emission from Stellar Winds

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1476587.svg)](https://doi.org/10.5281/zenodo.1476587)

This is a Python code to calculate the radio emission produced by the winds around stars. 
The code calculates thermal bremsstrahlung that is emitted from the wind, which depends directly on the density and temperature of the stellar wind plasma. 
The program takes input data in the form of an interpolated 3d grid of points (of the stellar wind) containing position, temperature and density data. 
From this it calculates the thermal free-free emission expected from the wind at a range of user-defined frequencies. 

This code is used in a paper accepted to Monthly Notices of the Royal Astronomical Society. Available at ArXiv: https://arxiv.org/pdf/1811.05356.pdf 

Please cite the above work if using this code.


## Installation
The code is available using pip:
`pip install radiowinds`

Or alternatively can be cloned directly from this repository.

### Dependencies
The calculations in this package depend on a number of different python packages, namely:
* **[numpy](http://www.numpy.org/)** 
* **[matplotlib](https://matplotlib.org/)** 
* **[pandas](https://pandas.pydata.org/)** 
* **[scipy](https://www.scipy.org/)** 
* [pytecplot](https://www.tecplot.com/docs/pytecplot/) 
* [moviepy](https://zulko.github.io/moviepy/install.html)
* [natsort](https://pypi.org/project/natsort/)

Required packages are shown in bold.

## Testing
The quickest way to test that the code is working is to use the test script included in the package.

To test:
```python
from radiowinds import test

#set up initial parameters for grid
ndim = 50
gridsize = 10
ordered = True

#Instantiate testcase class
t = test.testcase(ndim, gridsize, ordered)
#call the test
data = t.test()

```
The `data` variable should now contain an array of 3 variables: 2d array of intensity, radio flux in Jy, and the size of the radio photosphere in R<sub>&#8902;</sub>.

The above test will also output an image that should look like the following:

![Alt text](https://github.com/Dualta93/radiowinds/raw/master/radiowinds/test_ordered.png "Thermal Bremstrahlung from a stellar wind")

## Quick Example Code
To use this code with your own data follow the steps below.
You require that the data is in the format of an evenly interpolated 3D grid.

There are many ways to interpolate a 3d grid of points.
For the purposes of this example Tecplot was used to output an interpolated grid of points. 

The readData() function is used to get access to the data, it uses the pandas module. The radioEmission() function is the fastest way to make a calculation and get an output.
```python
import radiowinds.radio_emission as re

rstar = 1.05 #radius of star in R_sun

filename='/path/to/file.dat'
skiprows = 12 #this is the header size in the data file, which should be changed for users needs
ndim = 200 #This is the number of gridpoints in each dimension
gridsize = 10 #size of the radius of the grid in R_star

df = re.readData(filename, skiprows, ndim)

n = df[1] #grid density
T = df[2] #grid temperature
ds = df[0] #grid spacing along integration line
freq = 9e8 #observing frequency
distance = 10 #distance in parsecs

#remove density from behind star as this will not contribute to emisison
n = re.emptyBack(n, gridsize, ndim)

#find integration constant for grid of current size
int_c = re.integrationConstant(rstar)

I, sv, rv = re.radioEmission(ds, n, T, freq, distance, ndim, gridsize, int_c)
```
This should output an image of the intensity (and assign this data to `I`) from the wind and assign the radio flux to `sv` and the radio photopshere size to `rv`.

## Compute a Radio Spectrum
This repository also provides a way to automatically cycle through a range of frequencies to find the spectrum of a stellar wind.

This can be done by using the `spectrumCalculate()` function.

Continuing on from the quick example above:

```python
#set a range of frequencies to iterate over``
freqs = np.logspace(8,11,50)
output_dir = '/path/to/output' #where you want any output images to go
plotting=False #set to True to save images of intensity at each frequency to output_dir

svs, rvs = re.spectrumCalculate(output_dir, freqs, ds, n, T, d, ndim, gridsize, int_c, plotting=plotting)

```
`svs` will contain the flux in Jy at each frequency defined in freqs. To plot the spectrum simply use:
```python
plt.plot(freqs, svs)
```

### Creating animations
Using the images plotted from the spectrum function (provided `plotting == True`), one can use the [moviepy module](https://zulko.github.io/moviepy/) to make a short animation of the output.

```python
import radiowinds.make_animation as ma 

output_dir = '/path/to/output' #same directory as above

ma.make_animation(output_dir)
```
This will create an mp4 animation of the radio emission at different frequencies.

### Numerical Issues
Warning: 

If very low frequencies are used in the above calculations you run into some numerical problems.
Namely this is that the flux is overestimated. 

This is due to the fact that the optically thick region of the wind gets larger at lower frequencies,
and eventually will reach the boundary of the 3d grid. When this happens the code calculates the emission from denser
regions than would be visible had the 3d box been larger. 


## Functions
* ##### readData()
    This function allows you to read in the coordinates of the grid, the density of the grid and the temperature of the grid
    The pandas module is used to load data. 
    
    For the function to work properly, the datafile must be in column format with whitespace as the data seperator.
    The order of the data should be [X, Y, Z, density, temperature].
    
* ##### generateinterpolatedGrid()
    If Tecplot is installed on the system then this function should work to interpolated a 3d grid of data onto a regular 3d grid.
    It uses some function from the pytecplot module as well as some macro functions avaiable from Tecplot.
    As inputs it requires the .lay datafile, the number of points you want in your interpolated grid and the size of the grid in R<sub>&#8902;</sub>.
    The final input is the variable number of density and temperature as a list. e.g. [3, 20]
    ```python
    generateinterpolatedGrid('/path/to/tecplotfile.lay', 200, 10, [3, 20])
    ```
    Would create a grid of 200 x 200 x 200 points ranging from -10 to 10 R<sub>&#8902;</sub> in each direction (x,y,z).
    
    Ensure that pytecplot is installed on your system.

    In future might adapt this to work for the VisIT output.

### Author
Written by D&uacute;alta &Oacute; Fionnag&aacute;in in Trinity College Dublin, 2018
MIT License

Email : ofionnad@tcd.ie

[My Github](https://github.com/Dualta93)
