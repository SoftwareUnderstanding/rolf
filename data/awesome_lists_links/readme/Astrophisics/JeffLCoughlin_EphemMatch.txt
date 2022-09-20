# EphemMatch

The code reads in the period, epoch, positional information, etc. of all the Kepler DR25 TCEs, as well as the cumulative KOI list, and lists of EBs from the Kepler Eclipsing Binary Working Group (http://keplerebs.villanova.edu) as well as several catalogs of EBs known from ground-based surveys. The code will then perform matching to identify two different objects that have a statistically identical period and epoch (within some tolerance) and perform logic to identify which is the real source (the parent) and which is a false posivite due to contamination from the parent (a child).

For a very detailed description, see Section A.6 of Thompson et al. 2017/2018 (http://adsabs.harvard.edu/abs/2017arXiv171006758T) for the DR25 version, and read http://adsabs.harvard.edu/abs/2014AJ....147..119C which first used the technique on earlier KOI catalogs.


## Compiling and Running the Code

The EphemMatch code is provided, along with the required input files.

### Prerequisites

The code is written in C++ and only requires the standard C++ library (specifically, the required libraries are iomainip, iostream, fstream, cmath, cstdlib, ssstream, and vector). It has been tested to work with the g++ compiler, but should work with any standard C++ compile.

Optionally, a Gnuplot (http://www.gnuplot.info/) file is included (CCDPlot.gnu) to make nice PDF plots all of the matches on the focal plane / CCD array, using a file produced by the code (posfile.dat). So you will need Gnuplot with the pdfcairo terminal installed if you want to run the Gnuplot script.


### Compiling

To compile the code, use your available C++ compiler, and a recommended O2 level of optimization. For example:

```
g++ -Wno-unused-result -O2 -o match /home/jeff/KeplerJob/DR25/Code/DR25EphemMatch.cpp
```

### Running

To run EphemMatch, all filenames are hardcoded, so just type:

```
./match
```
Four files called TCEXXXMatches.txt, where XXX is either TCE, KOI, KEP, or GEB, which contain information on all matches that occur, including period, epochs, magnitudes, match significances, and CCD positions. A match (two objects) are shown by two lines - blank lines seperate matches. The file "AllMatchesSorted.txt" combines all this information into one file, sorted by period. "BestMatches.txt" and "BestMatchesSorted.txt" pick the best (most likely parent) match for each child, with the former sorted by KIC ID and the later sorted by period. The file "PeriodGroups.txt" contains all matched objects (parents and children, one per line) grouped accordingly such that each group should contain one parent and then one or more children. Groups are separated by a blank line. Finally, "Results.txt" contains the list of TCEs that are identified to be likely false positives, along with accompanying information such as the most likely parent object, spacial distances and magnitude differences between the two objects, matching significances, and the most likely mechanism of contamination.


## Citing EphemMatch

If using EphemMatch, please cite the following papers:

http://adsabs.harvard.edu/abs/2014AJ....147..119C

http://adsabs.harvard.edu/abs/2017arXiv171006758T


## Future Updates

Ideally this code would be re-written in Python, parallelzed, and generalized so it can be used for TESS and other surveys. I might one day. Hooray to anyone that writes their own though.

