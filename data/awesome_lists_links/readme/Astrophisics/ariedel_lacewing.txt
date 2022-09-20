# lacewing
Repository for the LACEwING moving group identification code (Riedel et al. 2017, 2017AJ....153...95R)

LocAting Constituent mEmbers In Nearby Groups (LACEwING) is an astrophysics code that uses the kinematics (positions and motions) of stars to determine if they are members of one of the 13 nearby young moving groups and 3 nearby open clusters. It considers membership in:
* epsilon Chameleontis
* eta Chameleontis (open cluster)
* TW Hydra
* beta Pic
* 32 Ori
* Octans
* Tuc-Hor
* Columba
* Carina
* Argus
* AB Doradus
* Carina-Near
* Coma Berenices (open cluster)
* Ursa Major
* chi01 Fornax
* The Hyades (open cluster)

It is written for Python 2.7 and depends upon Numpy, Scipy, and Astropy modules.

The basic operation is simple: 

1. Prepare a .csv file with a header containing the Name, RA, eRA, DEC, eDEC, pi, epi, rv, erv, pmra, epmra, pmdec, epmdec (the order does not matter) and fill in the columns with information about the stars of interest. Empty columns will be either ignored or errors estimated by the code.

RA and DEC are in decimal degrees; eRA and eDEC are in milliarcseconds (eRA is eRA * cos(DEC)), pi and epi are in milliarcseconds, rv and erv are in kilometers per second; pmra, pmdec, epmra, and epmdec are in milliarcseconds per year (where pmra and epmra are actually pmra * cos(DEC))

2. Run LACEwING:
python lacewing.py (filename) [(young?) (outputfilename) (verboseoutput?)]

 * If you do not specify any arguments, you will get a help message.

 * If you input a filename and nothing else, LACEwING will use field star parameters and regular output (in .csv format) will be sent to lacewing_output.csv

 * If you specify "young" as the third command-line argument, LACEwING will treat all the stars as if they are not ordinary field objects. This increases the estimated probabilities that the star is a member of one of the nearby young moving groups.

 * If you specify "verbose" as the fifth command-line argument, LACEwING will print out details information about how well the stars matched to each of the 16 groups, with information about how the individual proper motion, distance, RV, and spatial position matches turned out... which is useful for in-depth studies of why LACEwING predicted what it did. Verbose output is one line per group per star, rather than all the information on one line.



To run the LACEwING code itself requires three helper modules and one .csv file:
lacewing.py
kinematics.py
ellipse.py
astrometry.py
Moving_Groups_all.csv

More detailed instructions can be found in Riedel et al. (2017) 2017AJ....153...95R

----------------------------------------------------------------

This git repository also contains Moving_Group_all_prelim.csv, the calibration of LACEwING used in Riedel (2016) conference proceedings and Faherty et al. (2016). It is included for the purposes of reproducibility of results. To use it, rename the file to Moving_Group_all.csv

Its differences are:
* It considers the Pleiades open cluster (not within 100 pc) and the Hercules-Lyra moving group (does not seem to be real)
* It does not consider the Carina, Carina-Near, 32 Orionis or chi01 Fornax moving groups.
* Moving group parameters were determined from a slightly different set of bona-fide members, and sized according to an estimate of the total number of members rather than an actual calculation. The field population ratio was also calculated differently, and not present even at the 1:1 level in young star mode.
* The field star population was described by a single gaussian in UVW space rather than a multi-element population.

---------------------------------------------------------------

TRACEwING is an epicyclic traceback program that's designed to determine the distance between a star and the center of a moving group as a function of time going back into the past.

The theory is that all the stars in a moving group formed in the same place at roughly the same time, so running the clock back should put a true member very close to the center of the moving group, at the time the moving group formed.

Operation:
python tracewing.py (inputfile) (group_to_fit) (method) (min_age_of_group) (max_age_of_group) (ending_timestep) (number_of_monte_carlo_iterations)

Inputfile is a .csv file of the same format that LACEwING requires, which should contain kinematic information for all the stars you're interested in.

"Group_to_Fit" is the name of a moving group. which should have a file with a name like: Moving_group_<name>_epicyclic.dat in the same folder. That file should contain the ellipse parameters of the nearby young moving group as a function of time, and can be generated with tracewing_mgp.py

"Method" must currently be "epicyclic" to use the epicyclic traceback technique.

"Ending Timestep" must be the time in Myr ago you want the calculations to stop. (for instance, -40 for 40 Myr ago). It must be no farther back than the time you ran the moving group itself back to.

"Min Age of Group" and "Max Age of Group" are for the blue box that plots at the age range of the moving group (for instance, for beta Pic, Riedel et al. (in prep) used -12 and -25).

"Number of Monte Carlo Iterations" governs the accuracy of the traceback. The more traces, the more memory it takes. Unless you have a lot of memory, set this to 10000 or so.

TRACEwING requires one module and a set of .dat files (contained in TRACEwING_groups.tar.bz2) containing the ellipse parameters of the moving groups as a function of time.
tracewing.py
kinematics.py


-------------------------------------------------------------
lacewing_summary.py (filename)
will convert a file from verbose output format to compact format.
requires lacewing.py

lacewing_uvwxyz.py (filename) (XYZ?) (outfilename)
reads in a file and outputs three UVW phase-space plots for the stars in that file. If the second argument is XYZ, a second row of plots will contain XYZ plots for the star. It will generate the possible locations using ranges if data (RV and/or plx) does not exist.
2D ellipse parameters are found in the A2/B2/C2 (etc) columns of Moving_Group_all.csv
The program will also output a file of UVWXYZ parameters for all stars that have enough information to actually calculate UVWXYZ (or just XYZ).
Requires lacewing.py, ellipse.py, kinematics.py, and Moving_Group_all.csv

lacewing_mgpmaker.py (filename) (clustername) (iterations)
reads a list of stars, and generates the UVW and XYZ fitted ellipse parameters for them. The top line of the output file can be added to Moving_Group_all.csv (second line is uncertainties)

lacewing_mgpmaker_2d.py (filename) (clustername) (iterations)
like lacewing_mgpmaker, except instead of fitting 3D ellipsoids, it fits 2D ellipsoids to slices of data. The top line of the output file can be excerpted into Moving_Group_all.csv in the A2/B2/C2 columns. 

lacewing_montecarlo.py (iterations) (file number)
will 
1.) read stellar distributions from Moving_Group_all.csv
2.) generate <iterations> simulated stars within (and proportionately weighted according to) the distributions.
3.) run those stars through the lacewing algorithm and output the results, per cluster, into files named (clustername)(file number).

lacewing_percentages.py (filename) (young?) will run the actual calibrations on a single cluster file output from lacewing_montecarlo.py. It will output statistics plots and files of coefficients in a format ready to be pasted into Moving_Group_all.csv
If the second argument is "young", the program will ignore nearly all field stars (leaving only a 1:1 equal proportion of moving group and young field objects) when making and fitting its histograms. Filenames will have .youngonly in them.

