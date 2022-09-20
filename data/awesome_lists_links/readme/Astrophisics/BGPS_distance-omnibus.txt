distance-omnibus
================

### Description

This is the source repository for the Bolocam Galactic Plane Survey (BGPS) effort to resolve distance measurements to catalog sources. Through the Bayesian application of kinematic distance likelihoods derived from a Galactic rotation curve with prior Distance Probability Density Functions (DPDFs) derived from ancillary data, this code computes posterior DPDFs for catalog sources. This methodology and code base are generalized for use with any (sub-)millimeter survey of the Galactic plane. 

The methodology upon which **distance-omnibus** is based was introduced in [Ellsworth-Bowers et al. (2013, ApJ, 770, 39)](http://adsabs.harvard.edu/abs/2013ApJ...770...39E) and demonstrated on the BGPS version 1 data [(Aguirre et al. 2011, ApJS, 192, 4)](http://adsabs.harvard.edu/abs/2011ApJS..192....4A).  An expansion of the distance methodology to include a new kinematic distance likelihood and prior DPDFs is presented in [Ellsworth-Bowers et al. (2015, ApJ, 799, 29)](http://adsabs.harvard.edu/abs/2015ApJ...799...29E), and demonstrated on the re-reduced BGPS version 2 data of [Ginsburg et al. (2013, ApJS, 208, 14)](http://adsabs.harvard.edu/abs/2013ApJS..208...14G).  The DPDFs produced by **distance-omnibus** for the BGPS data are available through IPAC at [http://irsa.ipac.caltech.edu/data/BOLOCAM_GPS/distances/](http://irsa.ipac.caltech.edu/data/BOLOCAM_GPS/distances/).

The code is driven by a series of text configuration files that tell the code where to find survey data, contain physical parameters pertinent to the survey being used, and determine which DPDFs should be computed.  Once the `survey_info.conf` file has been properly populated and the all necessary data products have been obtained and their locations entered in the proper configuration files,  **distance-omnibus** runs autonomously, creating all necessary intermediate data products and computing posterior DPDFs.


=======
### Software Requirements

This package is written entirely in the [Interactive Data Language (IDL)](http://www.exelisvis.com/ProductsServices/IDL.aspx), and requires a recent version (8.0 or higher) to run.

Several external libraries of IDL routines are also required to run **distance-omnibus**.  These libraries must be installed on the local machine and their paths included in the IDL path.  The **distance-omnibus** code assumes you have a version of these libraries *no older* than the version current as of the release date shown below.
   * IDLASTRO (http://idlastro.gsfc.nasa.gov/) or (https://github.com/wlandsman/IDLAstro)
   * The Coyote Graphics System (http://www.idlcoyote.com/idldoc/cg/index.html) or (https://github.com/davidwfanning/idl-coyote)
   * The Markwardt IDL Library (http://www.physics.wisc.edu/~craigm/idl/)


=======
### Data Requirements

#### BGPS-Produced Data 

For the Eight-Micron Absorption Feature (EMAF) DPDF method, the distribution of Galactic mid-infrared emission, as described in [Ellsworth-Bowers et al. (2013)](http://adsabs.harvard.edu/abs/2013ApJ...770...39E) and computed from the model of  [Robitaille et al. (2012, A&A, 545, 39)](http://adsabs.harvard.edu/abs/2012A%26A...545A..39R), is required.  The version of the model used here is distributed as a FITS file, computed using the [Janus supercomputer](https://www.rc.colorado.edu/services/compute/janus), and may be found at [http://irsa.ipac.caltech.edu/data/BOLOCAM_GPS/distances/MW_model_ffore.fits](http://irsa.ipac.caltech.edu/data/BOLOCAM_GPS/distances/MW_model_ffore.fits).  This file contains the *foreground emission fraction* as a function of Galactic coordinates and heliocentric distance (*l,b,d*<sub>&#9737;</sub>).  The location of this file needs to be entered in the `local_layout.conf` configuration file.

Additionally, for the EMAF method, star-subtracted versions of the *Spitzer*/GLIMPSE IRAC Band 4 images (see below) are required.  It is infeasible for us to host a copy of these files (24 GB), but the code needed to produce them is included in this distribution.  Before running the **distance-omnibus** code proper, you must run the routine `omni_glimpse_starsub.pro`.  The star-subtraction process takes a considerable amount of time to run, so please plan accordingly.



#### Ancillary Data 

Because **distance-omnibus** estimates the distance to dense molecular cloud structures in the Milky Way based in part on ancillary data, the following publicly available data sets are required:
* The *Spitzer*/GLIMPSE mid-infrared survey V3.5 mosaics (available for the [GLIMPSE I](http://irsa.ipac.caltech.edu/data/SPITZER/GLIMPSE/images/I/1.2_mosaics_v3.5/) and [GLIMPSE II](http://irsa.ipac.caltech.edu/data/SPITZER/GLIMPSE/images/II/1.2_mosaics_v3.5/) coverage regions).  Specifically required are the Band 1 and Band 4 images (`*_I1.fits` and `*_I4.fits`).  (24 GB)
* The BU-FCRAO Galactic Ring Survey <sup>13</sup>CO(1-0) data cubes (available [here](http://grunt.bu.edu/grs-stitch/download-all.php)).  The code assumes you have the entire list of cubes to avoid edge effects.  (10 GB)


=======
### Release Information

Release version [v1.0.1](https://github.com/BGPS/distance-omnibus/archive/v1.0.1.tar.gz) available as of 01/13/15.

If your work makes use of **distance-omnibus**, please cite the following publications: [Ellsworth-Bowers et al. (2013, ApJ, 770, 39)](http://adsabs.harvard.edu/abs/2013ApJ...770...39E) and [Ellsworth-Bowers et al. (2015, ApJ, 799, 29)](http://adsabs.harvard.edu/abs/2015ApJ...799...29E).
