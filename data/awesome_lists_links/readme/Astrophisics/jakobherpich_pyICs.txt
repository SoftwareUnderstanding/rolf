Do you want to make isolated galaxy halo initial conditions? Then pyICs is made for you.

#pyICs

[pyICs](https://github.com/jakobherpich/pyICs) is a software for creating initial conditions (ICs) to simulate the formation of isolated galaxies. It was designed to create IC files in tipsy format (PKDGRAV/Gasoline/ChaNGa, successfully tested) but should also work for Gadget/Ramses/nchilada (all not tested) files.

[pyICs](https://github.com/jakobherpich/pyICs) depends heavily on the [pynbody](https://github.com/pynbody/pynbody) package as it uses pynbody to create the actual IC files.
Please install this package first.

##Getting started

If you have python configured with distutils the following should get you started:
```
$ git clone https://github.com/jakobherpich/pyICs.git
$ cd pyICs
$ python setup.py install
$ cd ..
$ python
>>> import pyICs
```

Check out the [GitHub page](http://jakobherpich.github.io/pyICs) for a quick guide on how to use
pyICs.

##Contributing
Help me improving pyICs. If you like to add additional functionality or improve the
code's performance you can create a fork of the repository, make the respective changes
and submit a pull request (see https://help.github.com/articles/using-pull-requests).

##Support and contact
If you found bugs or have feature requests you can
[submit an issue](https://github.com/jakobherpich/pyICs/issues).
If you need help using the code feel free to drop me an [email](mailto:jakob@jkherpich.de).

## Acknowledging pyICs
If you use pyICs for your work scientific work please mention it along with my name in the acknowledgments:
*This work made use of the open-source python initial condition creation package {\sc pyICs} written by Jakob Herpich (\url{https://github.com/jakobherpich/pyICs}).*

Additionally you can cite my paper (http://adsabs.harvard.edu/abs/2015arXiv151104442H) which is part of a series of papers in which pyICs was first used.

## Thanks
Special thanks go to Rok Roškar who gave me a great script to start out with and Stelios Kazantzidis who kindly provided his code for creating stable DM halos.
