# Least asymmetry

This is a python module for calculating the center of distributions using the
method of determining the point of least asymmetry. A description of the
algorithm can be found in the docs directory. The description is a stub of a
paper demonstrating its use for astrophysical measurements found [here](http://stacks.iop.org/1538-3873/126/i=946/a=1092}).

This module may either be used directly from a clone by running the ./build.sh
script to build the c++ extension, or by using the setup.py file to install the
module into the site packages directory. The actr routine itself is accessible
from the module itself, or from the asym.py file. As a side effect of how the
routine works, this module also provides centering by fitting a gaussian
(fitgaussian) or finding a weighted average (col) and can thus be used as a
general centering package

Module level Example:
```python
# create a demo image
import numpy as np
indy, indx = np.indices((30, 30))
gaus = a.gaussian(15, 14.5, 14.5, 3, 3, 0)
image = gaus(indy, indx)

noise = np.random.normal(0, 2, len(image.flatten()))
noise = noise.reshape(image.shape)
nimage = image+noise

# Use the routine
import least_asymmetry as a
center = a.actr(nimage, [14,14])[0]
```
