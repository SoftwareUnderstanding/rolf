# Tests of the Binary Offset Effect

This repository contains a Jupyter notebook with functions to test for the
binary offset effect in arbitrary CCD data. The details of this effect can be
found in [Boone et al. 2018.](http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1802.06914)

The code in this repository produces plots like Figure 6 in the paper, and can
be used to identify the binary offset effect in images from any detector. The
easiest input to work with is a dark or bias image that is spatially flat. The
code can also be run on images that are not spatially flat, assuming that there
is some model of the signal on the CCD that can be used to produce a residual
image.

The images from the various telescopes that were used to produce the examples
are not included in this repository due to space constraints, but can be
provided upon request. Most of the images were downloaded from public
repositories, and the observation IDs are provided.
