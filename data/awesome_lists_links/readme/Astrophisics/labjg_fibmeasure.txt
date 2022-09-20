# fibmeasure

This is a Cython implementation of various techniques to find the centre of
circular optical fibre images for astronomical fibre positioning feedback via
machine vision cameras. Specifically, it has been written during the design of
the WEAVE pick-and-place fibre positioner for the William Herschel Telescope.

Cython is used here to provide decent performance under Python, and to allow
easier porting to C or C++ in the future.

A makefile is included in the fibmeasure module folder, which should produce
'measure.so' using the Cython compiler. This is the only part of the module
that needs to be imported to do fibre measurements. See 'example.py' for usage.

You must have Cython, NumPy and SciPy. Also required for the cross-correlation
routines is the 'image_registration' module by Adam Ginsburg, available at
https://github.com/keflavich/image_registration
