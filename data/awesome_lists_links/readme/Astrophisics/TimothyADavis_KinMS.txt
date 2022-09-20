<b>Note: Active developement of KinMS has moved to Python - see [here](https://github.com/TimothyADavis/KinMSpy). This version is no longer developed. Use at your own risk.</b>

The KinMS (KINematic Molecular Simulation) package can be used to simulate observations of arbitary molecular/atomic cold gas distributions. The routines are written with flexibility in mind, and have been used in various different applications, including investigating the kinematics of molecular gas in early-type galaxies (Davis et al, MNRAS, Volume 429, Issue 1, p.534-555, 2013), and determining supermassive black-hole masses from CO interfermetric observations (Davis et al., Nature, 2013). They are also useful for creating input datacubes for further simulation in e.g. CASA's sim_observe tool.

This package should include KinMS.pro, makebeam.pro and rad_transfer.pro (the IDL astrolibrary is also required). It also includes a test suite, and two files associated with this in a separate folder.

The test suite contains various commented examples, which should show you some ways you can use the KinMS code. Each is explained in detail in the header to the procedure. Please open the KinMS_testsuite.pro file to view these, and try them.

If you find this software useful for your research, I would appreciate an acknowledgment to the use of the "KINematic Molecular Simulation (KinMS) routines of Davis et al., (2013)".

If you find any bugs, or wish to be kept up to date when new versions of this software are released, please email me at DavisT -at- cardiff.ac.uk

Many thanks,

Dr Timothy A. Davis
Cardiff, UK
