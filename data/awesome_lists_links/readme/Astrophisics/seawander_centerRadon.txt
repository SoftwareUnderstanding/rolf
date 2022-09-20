# Determination of Star Centers based on [Radon Transform](https://ui.adsabs.harvard.edu/#abs/2015ApJ...803...31P/abstract) [![DOI](https://zenodo.org/badge/47586602.svg)](https://zenodo.org/badge/latestdoi/47586602) <a href="http://ascl.net/1906.021"><img src="https://img.shields.io/badge/ascl-1906.021-blue.svg?colorB=262255" alt="ascl:1906.021" /></a>

This code is firstly used to determine the centers of the stars for the HST-STIS coronagraphic archive (Ren et al. [2017](http://adsabs.harvard.edu/abs/2017SPIE10400E..21R)).

## Installation
Checkout the code from this Github repository. Then open up a terminal window and navigate to the directory for this package. Run the following command to have an installation that will evolve with the development of this codebase.
```
$ python setup.py develop
```

## Running the code:
```python
import radonCenter
(x_cen, y_cen) = radonCenter.searchCenter(image, x_ctr_assign, y_ctr_assign, size_window = image.shape[0]/2)
```

### Inputs:
1. `image`: 2d array.
2. `x_ctr_assign`: the assigned x-center, or starting x-position; for STIS, the "CRPIX1" header is suggested.
3. `y_ctr_assign`: the assigned y-center, or starting y-position; for STIS, the "CRPIX2" header is suggested.
4. `size_window`: half width of the window to generate the cost function; for STIS, half the length of the image is suggested.


## References:
Pueyo, L., Soummer, R., Hoffmann, J., et al. 2015, [ApJ, 803, 31](https://ui.adsabs.harvard.edu/#abs/2015ApJ...803...31P/abstract)

Ren, B., Pueyo, L., Perrin, M. D., Debes, J. H, & Choquet, É. 2017, [Society of Photo-Optical Instrumentation Engineers (SPIE) Conference Series, 10400, 1040021](http://adsabs.harvard.edu/abs/2017SPIE10400E..21R)

