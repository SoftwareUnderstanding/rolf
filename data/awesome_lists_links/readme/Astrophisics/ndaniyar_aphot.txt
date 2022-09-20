A_phot computing library
========================

About
-----
  * This is a library for computing Photon Asymmetry (A_phot)
    parameter for morphological classification of clusters.
    This parameter is described in detail in the following publication:
    
        D. Nurgaliev, M. McDonald, B. A. Benson, C. W. Stubbs, A. Vikhlinin
        Robust Quantification of Galaxy Cluster Morphology Using Asymmetry and Central Concentration
        arXiv:1309.7044

  * This program is free software: you can redistribute it, modify
    it or include parts of it in your free or proprietary software, provided
    that you reference the aforementioned scientific publication.
    
This software is freely available, but we expect that all publications describing work using this software, or all commercial products using it, quote at least one of the references given below. This software is released under the BSD License

  * Copyright (C) 2013 Daniyar Nurgaliev



Prerequisites
-------------
  * Python 2.4+ 
  * Installed numpy, scipy and pyfits packages 



Installation
------------
  * Simply copy all python files (aphot.py, chandra.py, common.py, get_args.py, libaphot.py) to one folder.
  * aphot.py is the main program that parses the command line and calls all necessary library functions.
 


Examples
--------

  * Calculating A_phot given event 2 file, exposure map and point sources file:

        cd examples/SPT-CLJ0352-5647/
        python ../../aphot.py --evt_files=13490.evt.fits --exp_files=13490.exp.fits --reg_files=13490.psf.reg --ra=58.2412 --dec=-56.7985 --z=0.683 --R500=875.7

    Here "evt_files", "exp_files" and "reg_files" set the paths to Chandra level 2 event file, 
    "ra" and "dec" are the coordinates of the cluster, "z" and "R500" are cluster redshift and radius 
    corresponding to overdensity 500 relative to the *critical* density at the cluster redshift. 
    

  * Exposure maps and/or point sources can be omitted:

        python ../../aphot.py --evt_files=13490.evt.fits --ra=58.2412 --dec=-56.7985 --z=0.683 --R500=875.7

    In this case the program assumes uniform illumination of the chip, and uses an 
    internal ad-hoc routine for identifying point sources to be excluded from the image. This way of
    using the program is NOT recommended, because the assumption of uniform illumintation is far from
    reality, and the internal point sources finder is known to fail in some cases.
        

  * Instead of specifying all the parameters on the command line, they can be put in a configuration file:

        python ../../aphot.py --cfg=13490.cfg

    13490.cfg is a file with Python syntax which defines all relevant variables. 


  * Any variables from the configuration file can be overriden in the command line:

        python ../../aphot.py --cfg=13490.cfg --H=72

    Command line input has a higher priority and overrides values given in the configuration files.


  * Multiple observations can be used:
    
        python ../../aphot.py --cfg=all_obs.cfg 
        python ../../aphot.py --evt_files='13490.evt.fits 15571.evt.fits' --exp_files='13490.exp.fits 15571.exp.fits'  ...

    Event files, exposure maps, and point sources for both OBSIDs 13490 and 15571 will be used. 
    (See file all_obs.cfg for details.) 
    evt_files, exp_files, and reg_files can be either strings with filenames separated by spaces 
        
        evt_files = 'file1 file2'

    or Python lists
        
        evt_files = ['file1','file2']


  * Several event files can be used with one exposure map:

        cd examples/cl0405m4100
        python ../../aphot.py --evt_files='5756.evt.fits 7191.evt.fits' --exp_files=cl0405m4100.exp.fits --reg_files=cl0405m4100.reg \
                --ra=61.351 --dec=-41.004 --R500=742 --z=0.686


  * To check whether all input data are aligned correctly, add --inspect option:
    
        cd examples/SPT-CLJ0352-5647
        python ../../aphot.py --cfg=all_obs.cfg --inspect=CL

    This command will create several .png images with names starting with 'CL', that show events, exposures, 
    and points sources at different states of processing. Red shows exposure values, green - X-ray photons,
    blue - region used for background estimation, black - excluded point sources.


  * To suppress uncertainty calclation so that program runs faster, use

        python ../../aphot.py --cfg=all_obs.cfg --Nresamplings=0

    Nresamplings is the number of bootstrap iterations used to compute uncertainty. If it is set to 0, no bootstrapping happens.
    The default value is 100. It can be changed to find a balance between computation speed and accuracy of uncertainty estimation.



Configuration file variables
----------------------------

Variable | Meaning
---------|--------
H        | Hubble constant in km/s/Mpc, default value 70
Om       | Matter density (Flat cosmology is assumed, so that Dark Energy density = 1-Om), default value 0.27  
Eband    | X-ray energy band in eV, default value [500,5000]
R_bkgr   | Annnulus to extract background level, in units of R500, default value [2,4]
annuli   | Annuli for computing Aphot, default value [0.05, 0.12, 0.2, 0.3, 1.0]
Nresamplings |   Number of Monte-Carlo iterations for uncertainty estimation. If 0, uncertainty is not computed. Default value is 100


