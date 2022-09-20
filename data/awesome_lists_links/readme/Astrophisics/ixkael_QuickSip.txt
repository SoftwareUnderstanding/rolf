# QuickSip
Quickly projects Survey Image Properties (e.g. seeing, sky noise, airmass) into Healpix sky maps with flexible weighting schemes. Initially designed to produce observing condition "systematics" maps for the Dark Energy Survey (DES), but will work with any multi-epoch survey and images with valid WCS.

Tested against Mangle outputs; QuickSip will reproduce the Mangle magnitude limit maps at sub-percent accuracy but doesn't support additional masks (stars, trails, etc) in which case Mangle should be used. Thus, QuickSip can be seen as a simplified Mangle to project image properties into Healpix maps in a fast and more flexible manner.

For DES members, more details and products can be found at https://cdcvs.fnal.gov/redmine/projects/des-lss/wiki/Systematics_maps. Some examples below.

<img src="http://inspirehep.net/record/1384083/files/pics_ccdplot.png" height="200" />
<img src="http://inspirehep.net/record/1384083/files/pics_DES0009-4914_nside8192_oversamp4_count__total.png" height="200" />
<img src="http://inspirehep.net/record/1384083/files/pics_DES0009-4914_nside8192_oversamp4_FWHM__mean.png" height="200" />


# Need help?
Please use github issues or email (Boris.leistedt@gmail.com)! Tutorials and improvements are under way but will come faster if you trigger them.

# License
Copyright 2012-2016 Boris Leistedt.
QuickSip is free software made available under the MIT License. For details see the LICENSE file.

#Publications
The outputs of QuickSip are abundantly use within DES, but the main reference describe the code and its application to DES SV data is `Leistedt et al (2016, to be published in ApJS)
<http://arxiv.org/abs/1507.05647>`. Please cite this paper if you use QuickSip or use the ideas presented in this paper. The BibTeX is

    @article{Leistedt:2015kka,
          author         = "Leistedt, B. and others",
          title          = "{Mapping and Simulating Systematics Due to
                            Spatially-Varying Observing Conditions in DES Science
                            Verification Data}",
          collaboration  = "DES",
          year           = "2015",
          eprint         = "1507.05647",
          archivePrefix  = "arXiv",
          primaryClass   = "astro-ph.CO",
          reportNumber   = "FERMILAB-PUB-15-310-A-AE",
          SLACcitation   = "%%CITATION = ARXIV:1507.05647;%%"
    }

# Further remarks
## Requirements
Healpy, Pyfits, Numpy, Matplotlib.

## Limitations
- Python 2.7 only. But upgrade should be easy.
- Pixel size, coadd and WCS conventions hard-coded for DES/DECAM. But easy to adapt to other conventions.
- Projects into Healpix maps. But easy to generalize to other pixelizations.

## How to use?
Unfortunately there isn't a full tutorial (yet). But the script des_run.py is what we run for DES. The main input is a fits table containing all the relevant image properties, i.e. each row corresponds to a CCD and the columns are the essential WCS properties as well as the quantities and need to be projected/mapped. It is fairly clear from this script what these columns should be.

It should be fairly straightforward to modify this script to accommodate any input file formats and column name/conventions. Modifications to quicksip.py are needed to change the projection scheme, add a weighting scheme, change the WCS convention, or adapt the code to another survey.

## About the outputs
- Each HEALPix file is a cut-sky map, i.e. a FITS Table containing two arrays: the pixel indices (in ring format) and the values of the map at these pixels. These files can be read by pyfits in Python, or with the read_fits_cut4/mollview routines in IDL.
- The maps are averaged from super-sampled maps 2 resolutions higher, i.e. the value in each pixel at the base resolution (e.g., 4096) is averaged from 16 nested subpixels (e.g., at resolution 4*4096=16384). Unlike the HEALPIX ud_grade routines which degrade the holes and unseen pixels (yielding artifacts in low res maps), the unseen subpixels are ignored when averaging at high resolution. This gives a much better accuracy when using the maps in the context of cross correlation studies. Therefore, please do not degrade the maps, as it would affect their precision; you should rather directly download and exploit the resolution you need for your analysis.
- Magnitude limit maps are also available. They use the same formula as the Mangle code, and match the later at <1% accuracy when projected at the same resolution.
These maps are merely the result of projecting the CCDs on the sky, and performing mathematical operations on the systematics, nothing else. They do not incorporate any extra geometrical information, such as the tiling geometry (=the ccds are not truncated to fit within the tiles) and the mask (stars, bittrails, etc). Consequently, the maps are continuous and extend beyond the tiling coverage of SVA1. Therefore, for most applications (e.g., cross-correlations) you must combine these maps with the Mangle masks.
