# HRK: HII Region Kinematics
Generate simulated radio recombination line observations of HII regions with
various internal kinematic structure. Fit single Gaussians to each pixel of the
simulated observations and generate images of the fitted Gaussian center
and full-width half-maximum (FWHM) linewidth. Please reference https://doi.org/10.5281/zenodo.5205092.

## Installation
The easiest way to install this package is
```bash
pip install git+https://github.com/tvwenger/hii-region-kinematics.git
```

Alternatively, download the code and install:
```bash
git clone https://github.com/tvwenger/hii-region-kinematics.git; cd hii-region-kinematics; python setup.py install; cd ..
```

## Usage
The primary functionality is provided via the `kinematic-model.py` script, which is placed in your
path upon installation of this package.

```bash
kinematic-model.py --help
usage: kinematic_model.py [-h] [--outdir OUTDIR] [--density DENSITY] [--temperature TEMPERATURE] [--diameter DIAMETER]
                          [--distance DISTANCE] [--nonthermal NONTHERMAL] [--beam BEAM] [--noise NOISE] [--grid GRID]
                          [--nchan NCHAN] [--velwidth VELWIDTH] [--imagesize IMAGESIZE] [--rrl RRL] [--deltan DELTAN]
                          [--kinematic KINEMATIC [KINEMATIC ...]] [--overwrite]
                          modelname

HII Region Kinematic Models

positional arguments:
  modelname             Model name, added to saved FITS images

optional arguments:
  -h, --help            show this help message and exit
  --outdir OUTDIR       Output directory for FITS images (Default: current directory)
  --density DENSITY     Electron density (cm-3; default 250)
  --temperature TEMPERATURE
                        Electron temperature (K; default 8000)
  --diameter DIAMETER   Diameter (pc; default 2.0)
  --distance DISTANCE   Distance (kpc; default 5.0)
  --nonthermal NONTHERMAL
                        Non-thermal FWHM line width (km/s; default 15.0)
  --beam BEAM           Beam FWHM width (arcsec; default 90.0)
  --noise NOISE         Pre-convolution image noise (mJy/arcsec2; default 0.001)
  --grid GRID           Number of points along model grid side (default 128)
  --nchan NCHAN         Number of velocity channels (default 256)
  --velwidth VELWIDTH   Full velocity range, centered at 0 (km/s; default 200.0)
  --imagesize IMAGESIZE
                        Image width (arcsec; default 600.0)
  --rrl RRL             RRL principal quantum number (default 85)
  --deltan DELTAN       RRL principal quantum number transition (default 1)
  --kinematic KINEMATIC [KINEMATIC ...]
                        
                        Add a kinematic component. Kinematics are applied
                        in the order passed, like:
                        --kinematic type1 <args1> --kinematic type2 <args2>
                        
                        Options:
                        --kinematic solidbody <eq_speed> <sky_pa> <los_pa>
                        where <eq_speed> is equatorial rotation speed (km/s)
                        <sky_pa> and <los_pa> define angular momentum vector (deg)
                        
                        --kinematic differential <eq_speed> <r_power> <sky_pa> <los_pa>
                        where <eq_speed> is equatorial rotation speed at surface (km/s)
                        <r_power> is exponent of radial change in rotation speed
                        <sky_pa> and <los_pa> define angular momentum vector (deg)
                        
                        --kinematic outflow <speed> <angle> <sky_pa> <los_pa>
                        where <speed> is radial outflow speed (km/s)
                        <angle> is opening angle (deg)
                        <sky_pa> and <los_pa> define positive velocity outflow axis

                        --kinematic expansion <alpha> <beta>
                        where <alpha> is the expansion speed at the surface (km/s)
                        and <beta> is the exponential of the power law

  --overwrite           Overwrite existing FITS images
```

## Model Results
The model results are saved to several FITS images:
1. `<modelname>_true.fits` The model sky brightness distribution as a function of velocity.
2. `<modelname>_velocity.fits` The three-dimensional nebula velocity distribution.
3. `<modelname>_obs.fits` The observed brightness distribution as a function of velocity.
4. `<modelname>_fit_center.fits` The center velocity of the fitted Gaussian in each pixel.
5. `<modelname>_fit_e_center.fits` The center velocity uncertainty of the fitted Gaussian in each pixel.
6. `<modelname>_fit_fwhm.fits` The FWHM line width of the fitted Gaussian in each pixel.
7. `<modelname>_fit_e_fwhm.fits` The FWHM line width uncertainty of the fitted Gaussian in each pixel.

## Example
See the `example` directory for an example call of the script as well as the resulting images.

## Issues and Contributing

Please submit issues or contribute to the development via [Github](https://github.com/tvwenger/hii-region-kinematics).

### License and Warranty

GNU Public License
http://www.gnu.org/licenses/

This package is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This package is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this package. If not, see http://www.gnu.org/licenses/
