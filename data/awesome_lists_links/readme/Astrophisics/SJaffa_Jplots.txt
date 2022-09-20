## J plots and J3D

We can use the principal moments of inertia of a shape to
classify astronomically ineresting structures in 2D (J plots) and 3D (J3D).

# J plots (2D)

This code is able to separate centrally concentrated structures
(cores), elongated structures (filaments) and hollow circular
structures (bubbles) from the main population of ‘slightly
irregular blobs’ that make up most astronomical images.

This can be applied to any 2D greyscale pixelated image (single
wavelength/tracer or column density).

Examples of the usage of J plots are given in the tests folder.

A full description of this algorithm, the proof of concept 
tests, and example astronomical applications are described 
in the paper. A PDF is included in the docs folder or can be 
found on arXiv at https://arxiv.org/abs/1803.01640

# J3D (3D)

This code is able to separate centrally concentrated structures, 
elongated structures (filaments), hollow structures, and prolate/oblate 
spheroids.

This can be applied to any 3D greyscale data cube (single
wavelength/tracer or column density in PPP or PPV).

It should be pretty simple to run this code and get the plots 
out, but if you want to do anything more complicated please
contact the authors on s.jaffa@herts.ac.uk.

A full description of this algorithm, the proof of concept 
tests, and some example astronomical applications are described 
in the paper which is currently in preparation.

# Installing and running

This code is written in Python 3 (so should also run in Python 2). It requires the following libraries:

- numpy
- scipy
- matplotlib
- astropy (optional, for reading fits files in example scripts)
- astrodendro (optional, for image segmentation in example scripts)

The scripts included in the 'tests' folder give basic examples of how to use J plots and J3D on common types of astronomical data. The input and output files are included so you can test the code yourself, or modify these examples to analyse your own data.

# Editing and contributing

If you wish to use the code as it is to analyse your own work, please cite the relevant papers in your publication. If you wish to edit the code, expand it with new features, point out bugs or suggest improvements please contact the authors either by email or through the GitHub issue tracking page. This code is shared under the GNU GPLv3 license. For further detail please see the LICENSE file.


# Citation

If you use this code in your work, please cite the original [J plots paper](https://arxiv.org/abs/1803.01640) and/or cite the code directly via the [Astrophysical Source Code Library](https://ascl.net/code/v/2657). If you need help running the code or interpreting your results, we would be happy to collaborate!
