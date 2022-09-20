# Astro-Toyz

Astro-Toyz is an astronomy package meant to be an addon to the [Toyz](https://github.com/fred3m/toyz)
web framework, which allows researchers and classrooms to visualize and interact with data
stored on a remote server.

Currently the main additions of the astrotoyz package are:

1. An astro-viewer tile, which is built on top of the Toyz image viewer but contains additional support for WCS, selecing regions of a plot, and other astro-specific tools
2. Support for astropy Tables as a Data Source

In the future the goal is to make Astro-Toyz a front-end for astropy, allowing students to take
advantage of data reduction tools with limited to no programming necessary.

##Installation
Copy or clone the astro-toyz repository to your local machine

    git clone https://github.com/fred3m/astro-toyz .

Then enter the 'astro-toyz' directory and run

    python setup.py install

to install astro-toyz. If you want to install all of required and optional dependencies you can run:

    pip install -e .[all]

To just install the mandatory dependencies run

    pip install -e .[base]

## Development
Astro-Toyz is still undergoing active development and in the future will contain numerous additional
features, including displaying source catalogs on the image viewer.