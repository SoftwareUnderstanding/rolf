The TesseRACt package is designed to compute concentrations of simulated dark
matter halos from volume info for particles generated using Voronoi tesselation.
This technique is advantageous as it is non-parametric, does not assume 
spherical symmetry, and allows for the presence of substructure. For a more
complete description of this technique including a comparison to other 
techniques for calculating concentration, please see the accompanying paper 
[Lang et al. (2015)](http://arxiv.org/abs/1504.04307).

This package includes:

 * **vorovol**: C program for computing the Voronoi diagram of particle data in 
    a number of formats including Gadget-2, Gasoline, binary, and ASCII as well
    as BGC2 halo catalogues.
 * routines for compiling, running, and parsing **vorovol** output
 * routines for computing concentrations using particles volumes, traditional 
    fitting to an NFW profile, and non-parametric techniques that assume 
    spherical symmetry.
 * routines and test halos for running many of the performance tests presented in 
    [Lang et al. (2015)](http://arxiv.org/abs/1504.04307).

Below are some useful links associated with TesseRACt:

 * [PyPI](https://pypi.python.org/pypi/tesseract) - The most recent stable release.
 * [Docs](http://pytesseract.readthedocs.org/en/latest/) - Tutorials and descriptions of the package modules and functions.
 * [Lang et al. (2015)](http://arxiv.org/abs/1504.04307) - The accompanying scientific paper.

If you would like more information about TesseRACt, please contact [Meagan Lang](mailto:cfh5058@gmail.com).