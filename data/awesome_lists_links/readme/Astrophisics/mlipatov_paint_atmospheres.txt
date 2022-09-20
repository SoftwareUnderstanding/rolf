<img src="logo.png" width="20%">

# Paint the Atmospheres of Rotating Stars (PARS)

This software quickly computes magnitudes and spectra of rotating stellar models. The models incorporate **Roche mass distribution** (where all mass is at the center of the star), **solid body rotation**, and **collinearity of effective gravity and energy flux**.

## Getting Started

The following instructions describe how to install PARS on macOS 10.15.4. These may have to be modified for other operating systems.

### Prerequisites

* git
* python

### Installation

#### Stable version

Go to the web page with the [latest release](https://github.com/mlipatov/paint_atmospheres/releases/latest), download the source code as a tar.gz file, put the file in the directory where you want to install PARS.

Un-compress the file, go to the software's top directory and install PARS.

```
tar -xf paint_atmospheres-x.x.x.tar.gz
cd paint_atmospheres-x.x.x
pip install .
```

#### Current version

Go to the directory where you want to install PARS, clone it in that directory, go to the software's top directory and install PARS.

```
cd <directory name>
git clone https://github.com/mlipatov/paint_atmospheres
cd paint_atmospheres
pip install .
```

### Setup and checks

Place a file with limb darkening information from atmosphere models in the data directory.
```
cp ~/im01k2.pck ./data
```

Print the command line syntax for the executables.
```
calc_limbdark -h
calc_star -h
calc_spectra -h
```

## Usage

The following instructions describe how to access the functionality of PARS in several different ways.

### Compute a Spectrum

Compute fits of intensity versus viewing angle from the limb darkening information.
```
calc_limbdark data/im01k2.pck data/limbdark_m01.pkl 0.1 0.4
```

Perform the inclination-independent computations for a stellar model.
```
calc_star 'data/limbdark_m01.pkl' 'data/vega.pkl' 0.6151 40.346 2.165 2.815 2.3694e19 100
```

Perform the inclination-dependent computations.
```
mkdir data/vega
calc_spectra 'data/vega.pkl' 'data/vega/' -i 0.088418
```

Look at the resulting spectrum.
```
cat data/vega/*.txt | more
```

### Compute Magnitudes

Move a file with a filter transmission curve into the filter directory.

```
mv data/Generic_Bessell.V.dat data/filters/
```

Compute fits of *filtered* intensity versus the cosine of the viewing angle, then follow the same steps as above.
```
calc_limbdark data/im01k2.pck data/limbdark_m01f.pkl 0.1 0.4 -f data/filters/
calc_star 'data/limbdark_m01.pkl' 'data/vega.pkl' 0.6151 40.346 2.165 2.815 2.3694e19 100
calc_spectra 'data/vega.pkl' 'data/vega/' -i 0.088418
cat data/vega/*.txt | more
```

### Run Sample Scripts

Go to the directory with scripts, run a script, look at the result.
```
cd pa/usr/
python 09_colormag_inclinations.py
ls ../../vega_colormag.pdf
```

#### Scripts

These create figures 3 - 10 in [LB] (see References below).

* Intensity vs. viewing angle, goodness of fit checks: [1](pa/usr/03a_Imu_fits_min.py) followed by [2](03b_Imu_fits.py)
* [Temperature error](pa/usr/04_temperature.py)
* Error due to interpolation in [temperature](pa/usr/05a_temperature_interpolation.py) and [gravity](pa/usr/05b_gravity_interpolation.py)
* [Convergence of the longitudinal integral](pa/usr/06_convergence.py)
* [Error in the longitudinal integral](pa/usr/07_error_heat_map.py)
* [Comparison with an observed spectrum](pa/usr/08_vega_spectrum_comparison.py)
* [Color-magnitude diagram for a range of inclinations](pa/usr/09_colormag_inclinations.py)
* [Planetary transits](pa/usr/10_transit.py)

### Extra: Compute Spectra of Brown Dwarfs

Unzip the file with brown dwarf plane-parallel atmospheres in CSV file format.
```
cd paint_atmospheres/data/
tar -xf bd_atmospheres.zip
```

Compute fits of intensity versus viewing angle from the limb darkening information.
```
python ../pa/usr/brown_dwarfs/calc_limbdark_bd.py
```

#### Figure 1: synthetic spectra of rotating brown dwarf models and ratios of spectra

Compute the spectra of the T7 dwarf in Tannock et al. (2021) for the two extreme inclinations; plot.
```
python calc_bd.py bd_pkl/J0348-6022_few/ bd_spectra/J0348-6022_few/ 4.3e-6 3 9 -o 0.42
plot_bd.py ../../../data/bd_spectra/J0348-6022_few/ -d 10 --ratio
```

#### Figure 2: synthetic spectra of the L3.5 dwarf from Tannock et al. and of beta Pictoris B

Calculate and plot the spectrum of the rapidly rotating L3.5 BD (note that the cloud-free Sonora atmospheres that we use are technically not appropriate for objects this hot).
```
calc_star data/limbdark_BD_00.pkl data/bd_pkl/J0407+1546.pkl 0.324443 0.0000977095 0.064 0.1 3.086e19 100
calc_spectra data/bd_pkl/J0407+1546.pkl data/bd_spectra/J0407+1546/ -i 0.000 1.5707963267948966 2
python plot_bd.py ../../../data/bd_spectra/J0407+1546/ -d 10
```

Calculate and plot the spectrum of Beta Pictoris b at 10 pc and two inclinations (luminosity is calculated from equatorial radius, not average).
```
calc_star data/limbdark_BD_00.pkl data/bd_pkl/betapicb.pkl 0.244337 0.000173182 0.0123 0.15 3.086e19 100
calc_spectra data/bd_pkl/betapicb.pkl data/bd_spectra/betapicb/ -i 0 1.5707963267948966 2
python plot_bd.py ../../../data/bd_spectra/betapicb/ -d 10
```

#### Figure 3: RMSD as a function of inclination and rotational speed, and
#### Figure 4: flux anisotropy ratio versus inclination and rotational speed

Calculate the models and the spectra.
```
python calc_bd.py bd_pkl/J0348-6022_400/ bd_spectra/J0348-6022_400/ 1.8e-7 11 21 -o 0.1 0.5 4
python calc_bd.py bd_pkl/J0348-6022_600/ bd_spectra/J0348-6022_600/ 9.3e-7 11 21 -o 0.1 0.5 4
python calc_bd.py bd_pkl/J0348-6022_880/ bd_spectra/J0348-6022_880/ 4.3e-6 11 25 -o 0.1 0.5 7
python calc_bd.py bd_pkl/J0348-6022_1500/ bd_spectra/J0348-6022_1500/ 3.6e-5 11 47 -o 0.1 0.5 4
python calc_bd.py bd_pkl/J0348-6022_2200/ bd_spectra/J0348-6022_2200/ 1.7e-4 11 43 -o 0.1 0.5 4
```

Plot.
```
python plot_T7.py 400
python plot_T7.py 600
python plot_T7.py 880
python plot_T7.py 1500
python plot_T7.py 880 # now that all the anisotropy ratios and rmsd values have been calculated
```

#### Movie: make a movie of the T7 dwarf spectra at different inclinations
```
calc_spectra data/bd_pkl/J0348-6022/rotating/J0348-6022.pkl data/bd_spectra/J0348-6022_movie/txt/ -i 0.000 1.5707963267948966 150
python plot_bd_inclinations.py ../../../data/bd_spectra/J0348-6022_movie/txt ../../../data/bd_spectra/J0348-6022_movie/jpg 6000 -t 
cd ../../../data/bd_spectra/J0348-6022_movie/
ffmpeg -framerate 10 -pattern_type glob -i '*.jpeg' -c:v libx264 -pix_fmt yuv420p -vf pad="width=ceil(iw/2)*2:height=ceil(ih/2)*2" ../J0348-6022_movie.mp4
```

## Authors

* Timothy D. Brandt
* [Mikhail Lipatov](https://github.com/mlipatov/)

## References 
	
* Lipatov M & Brandt TD, arXiv:2007.12779 [LB]
* Espinosa Lara F & Rieutord M (2011), A&A, 533, A43
* Castelli F & Kurucz RL (2004), arXiv:astro-ph/0405087 
* [Wolfram World: Cubic Formula](http://mathworld.wolfram.com/CubicFormula.html)
* [Wikipedia: Bilinear interpolation: Algorithm](https://en.wikipedia.org/wiki/Bilinear_interpolation#Algorithm)
* [Wikipedia: Newton's Method](https://en.wikipedia.org/wiki/Newton%27s_method)
* Press WH et al, Numerical Recipes, 3rd ed. (2007) 

## Cite PARS

If you use PARS in your work, please cite [the article that describes it](https://arxiv.org/abs/2007.12779). Thank you!
