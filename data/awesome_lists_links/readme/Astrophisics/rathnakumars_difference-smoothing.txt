Overview of folder contents
---------------------------
"codes": MATLAB codes implementing difference-smoothing algorithm

"crash_testing": Results from running the codes on rung3 and rung4 simulated light curves from TDC1

"time_delay_challenge": Simulated data and truth files from TDC0 and TDC1

Codes to run
------------
"display_lightcurves.m": Plot the light curves to measure time delay from

"find_time_delay.m": Measure the time delay between the light curves

"simple_uncertainty.m": Estimate "simple" uncertainty of the measured time delay and apply a correction to the measured delay

"comprehensive_uncertainty.m": Estimate "comprehensive" uncertainty of the measured time delay and apply a correction to the measured delay

"compute_tdc_metrics.m": Compute TDC performance metrics

Overview of "crash_testing" folder
----------------------------------
"results" files: Summary displayed after running "simple_uncertainty.m" is saved here

"truth_compare" files: One-line summary displayed after running "simple_uncertainty.m" is manually pasted here (along with the filename and true delay) to compare measured delay and true delay 
                       
"comprehensive_results" files: Summary displayed after running "comprehensive_uncertainty.m" is saved here

"interesting_cases" files: One-line summary displayed after running "comprehensive_uncertainty.m" is manually pasted here (along with the filename and true delay) to compare measured delay and true delay

Commands to initiate parallel pool in MATLAB 2015/2016
------------------------------------------------------
c = parcluster('local')

c.NumWorkers = < no. of physical cores to use >

parpool(c, c.NumWorkers)

Contact
-------
Email your queries to 'rathna@aries.res.in'
