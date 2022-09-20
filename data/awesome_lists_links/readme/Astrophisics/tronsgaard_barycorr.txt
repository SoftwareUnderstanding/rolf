# barycorr.py

Python routines that query Jason Eastman's web applets for barycentric
velocity and time correction (
[`barycorr.pro`](http://astroutils.astronomy.ohio-state.edu/exofast/pro/exofast/bary/barycorr.pro), 
[`utc2bjd.pro`](http://astroutils.astronomy.ohio-state.edu/time/pro/utc2bjd.pro) and 
[`bjd2utc.pro`](http://astroutils.astronomy.ohio-state.edu/time/pro/bjd2utc.pro))

When using one of the services provided through this module, please cite the
corresponding paper:

- [Wright and Eastman (2014), PASP 126, pp. 838–852](http://adsabs.harvard.edu/abs/2014PASP..126..838W) [BVC calculation]
- [Eastman et al. (2010), PASP 122, pp. 935–946](http://adsabs.harvard.edu/abs/2010PASP..122..935E) [BJD calculations]

The Python interface is written by René Tronsgaard (Aarhus University) and may
be used, modified or redistributed without restrictions.

See also: 
- http://astroutils.astronomy.ohio-state.edu/exofast/barycorr.html
- http://astroutils.astronomy.ohio-state.edu/time/utc2bjd.html
- http://astroutils.astronomy.ohio-state.edu/time/bjd2utc.html

## Installation

Download barycorr.py to a directory of choice.

The following packages are required (available in PyPI):
- [`numpy`](http://www.numpy.org/)
- [`requests`](http://python-requests.org)

Recommended:

- [`requests-cache`](https://pypi.org/project/requests-cache/)

## Usage example

```python
import barycorr
params = {
    'jd_utc': [2457535.067362, 2457462.12724721],
    'ra': 293.08995940,
    'dec':  69.66117649,
    'lat': 28.2983,
    'lon': -16.5094,
    'elevation': 2400,
    'pmra': 598.07,
    'pmdec': -1738.40,
    'parallax': 173.77,
    'rv': 26780,
    'zmeas': [-4.99432219e-06,  1.16637407e-05]
}
barycorr.bvc(**params)

# Returns: numpy.array([-1312.08186269,   515.87479325])
```

## Version history
##### Version 1.4 (16 September 2021)
Added unit test and prevent the code from crashing when there is a warning about outdated leap second file.
##### Version 1.3 (16 June 2020)
Added cache functionality (courtesy of @vterron)
##### Version 1.2 (01 Oct 2017)
Allowed keyword `raunits` to switch between hours and degrees (courtesy of Mathias Zechmeister)
##### Version 1.1 (26 May 2016)
Initial release
