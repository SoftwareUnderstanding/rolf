# Jet Curry

Models the 3D geometry of AGN jets.

JetCurry, Copyright 2017. Katie Kosak. Eric Perlman. Copyright Details: GPL.txt

Last Edited: March 2018. Questions: Contact Katie Kosak,  [katie.kosak@gmail.com](mailto:katie.kosak@gmail.com)


## Usage

python JetCurryMain.py input [-out_dir] 

**Required arguments**

**input**: a single FITS file or directory (full or relative path) 

**Optional arguments** 

**-out\_dir**: full or relative path to save output files.  Output directory is created if it doesn't exist. Default output is current working directory if -out\_dir is not specified. 

**Example**
> python JetCurryMain.py ./KnotD\_Radio.fits

>  python JetCurryMain.py ./data -out_dir /foo/bar/

## Notes

A GUI will display the FITS image. Click on the image with your left mouse button to choose the upstream bounds starting point and right click to choose the downstream bounds. You may continuously click on the image until you are satisifed with the regions of interest. These values can also be entered by typing. Once satisifed, click the Run button to process the data.

Data products are organized by the FITS filename. For example, if the output directory is /foo/bar and the filename is KnotD_Radio.fits, then data products will be saved to /foo/bar/KnotD_Radio. 

## TODO

Optimize with cython or numba.
