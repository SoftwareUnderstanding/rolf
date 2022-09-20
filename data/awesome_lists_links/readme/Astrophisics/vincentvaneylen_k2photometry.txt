# k2photometry
This code can be used to read, reduce and detrend K2 photometry. 

This code is accompanied by a paper (Van Eylen et al. 2015, ApJ), "THE K2-ESPRINT PROJECT II: SPECTROSCOPIC FOLLOW-UP OF THREE EXOPLANET SYSTEMS FROM CAMPAIGN 1 OF K2". 
Feel free to reuse and modify this code or any part of it. Please reference this publication if you found this K2 pipeline useful. Contact vincent@phys.au.dk for more information.

The input is pixel files which can be downloaded from the MAST database (https://archive.stsci.edu/k2/data_search/search.php). 
The output includes raw lightcurves, detrended lightcurves and a transit search can be performed as well. Stellar variability is not typically well-preserved but parameters can be tweaked to change that.

The main file is called run_pipeline.py. An example how to use it, with a pixel file included in folder example_input, is given in example.py.


Disclaimer: The BLS algorithm used to detect periodic events is taken from a Python implementation by Ruth Angus and Dan Foreman-Mackey (see https://github.com/dfm/python-bls), and repeated here. A transit model using Mandel & Agol models is included for completeness but not a part of the pipeline, it was implented in Python by Ian Crossfield (http://www.lpl.arizona.edu/~ianc/python/transit.html).

