# Single-Pulse Event Group IDentification (SPEGID)

Single-Pulse Event Group IDentification (SPEGID) algorithm identifies astrophysical pulse candidates as trial single-pulse event groups (SPEGs) by first applying Density Based Spatial Clustering of Applications with Noise (DBSCAN) (Ester et al. 1996) on trial single-pulse events and then merging the clusters that fall within the expected DM (Dispersion Measure) and time span of astrophysical pulses (Cordes & McLaughlin 2003). SPEGID also calculates the peak score for each SPEG in the S/N versus DM space to identify the expected peak-like shape in the signal-to-noise (S/N) ratio versus DM curve of astrophysical pulses. Additionally, SPEGID groups SPEGs that appear at a consistent DM and therefore are likely emitted from the same source.

After running SPEGID, periocity.py can be used to find (or verify) the underlying periodicity among a group of SPEGs (i.e., astrophysical pulse candidates).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

For each beam (i.e., independent observation of a specific sky position), the single-pulse search data processed by the PRESTO code single_pulse_search.py have to preprocessed to into the following two files: MJD/beam_name/beam_namesinglepulses.csv and MJD/beam_name/beam_name_inf.txt, as the example provided in /test_data. 
```
56475/p2030.20130702.G33.79+00.82.N.b3.00000/p2030.20130702.G33.79+00.82.N.b3.00000singlepulses.csv 
56475/p2030.20130702.G33.79+00.82.N.b3.00000/p2030.20130702.G33.79+00.82.N.b3.00000_inf.txt
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests
To run SPEGID,
```
cd 56475/p2030.20130702.G33.79+00.82.N.b3.00000
python SPEGID.py
```
to run periodicity.py (after running SPEGID.py),
```
cd 56475/p2030.20130702.G33.79+00.82.N.b3.00000
python periodicity.py
```

### Output

The output of SPEGID.py is single-pulse event groups (SPEGs) with a list of features.
The output of periodicity.py is a list of underlying periodicity found among SPEGs.

## For Automatic classification of SPEGs Using Supervised Machine Learning

Please refer to:
* Di Pang, Katerina Goseva-Popstojanova, Thomas Devine, Maura McLaughlin, A novel single-pulse search approach to detection of dispersed radio pulses using clustering and supervised machine learning, Monthly Notices of the Royal Astronomical Society, Volume 480, Issue 3, November 2018, Pages 3302–3323, https://doi.org/10.1093/mnras/sty1992

## Built With

* SPEGID is written in Python 2.7.
* Implementation SPEGID in Python 3 is added.

## Authors

Di Pang, Katerina Goseva-Popstojanova, Maura McLaughlin and Thomas Devine.  

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References
* Ester M., Kriegel H. P., Sander J., Xu X., 1996, Second International Conference on Knowledge Discovery and Data Mining, pp 226–231
* Cordes J. M., McLaughlin M. A., 2003, The Astrophysical Journal, 596, 1142

## Acknowledgments

* This work is partially supported by the National Science Foundation under Award No. OIA-1458952. 


