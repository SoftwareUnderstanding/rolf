## tl;dr
This package provides a fast and reliable way to test redshifts found from sub-mm redshift searches. The accompanying paper (Bakx & Dannerbauer et al. in prep) will provide adequate discussion, while this page provides the code used to make all graphs in the paper. These scripts can be adapted as necessary by the user for their needs, which include:

1. Graphically test the robustness of a spectroscopic redshift of a galaxy (Fig_1.py, Fig_2.py, Fig_3.py, and Fig_4.py)
2. Test the efficiency of an instrument towards spectroscopic redshift searches (tuningGraph.py and tuningGraph_individual.py)
3. Optimize observations of tunable institutes (such as ALMA) for upcoming redshift searches (tuningOptimization.py)

The code can be used to readily recreate the figures from Bakx & Dannerbauer in prep. For example, the following command produces Figure 1 in the paper:
'''
python Fig_1.py
'''

## Requirements
- Python 3.8
- numpy
- matplotlib

## RSG package
### RSGplot(filter_down,filter_up,IDname,z_phot=-99,sl_freq_obs=[-99],figSizeX=6,figSizeY=4,redshift_down=0,redshift_up=7,single_line_colour='#FFD79F',multi_line_colour='#9DDBFF',LSBUSB=False,DPIVal=400,nr_of_CO_lines = 20,dzUncertainty=0.13)
Produces the images shown in Fig. 1 through 4.
- filter_down: a list-type indicating the lower-frequency component of the bandwidth (i.e., filter_down = [90,125])
- filter_down: a list-type indicating the upper-frequency component of the bandwidth (i.e., filter_up = [100,130])
- IDname: Used only for the file name that will be produced
- z_phot: Can be set to include the phot-z image, assuming an error of dz / (1+z_spec) ~ 0.13. 
- sl_freq_obs: list-type of the observed spectral lines
- figSizeX, figSizeY: the image size parameters
- redshift_down, redshift_up: the lower- and upper-redshift bounds on the figure
- single_line_colour, multi_line_colour: the colours of the image can be adjusted by the user
- LSBUSB: Are the filters correlated (as is typical for heterodyne instruments?)
- DPIVal: set the DPI of the observations
- nr_of_CO_lines: The number of CO lines you wish to plot
- dzUncertainty: Defined as (delta z)/(1+z), influencing the photometric redshift bullet.


### giveALMA(band, lower_freq_ratio)
Shows the spws for ALMA given the band and the lower_freq_ratio. the lower_freq_ratio is a value between 0 and 1, where 0 indicates the lowest possible frequency-combination of ALMA, and 1 indicates the highest possible frequency-combination of ALMA.

### tauCalculator(PWV, printTau='False')
Set the percipitable water vapour (PWV), and produces the interpolated value from the PWV. Sourced from the APEX website: https://www.apex-telescope.org/weather/RadioMeter/index.php

### RSGquality(filter_down,filter_up,redshift_array,includeCI = False, nr_of_CO_lines=20,lin_arr_size=10000,sigma_threshold=5,dzUncertainty=0.13)
Produces a list of 'quality' metrics of a proposed redshift search, given a specific bandwidth and redshift distribution. This can include CI lines. 
[no_lines,one_line,two_lines,more_lines,robust_single_lines,non_robust_double_lines]
- filter_down: a list-type indicating the lower-frequency component of the bandwidth (i.e., filter_down = [90,125])
- filter_down: a list-type indicating the upper-frequency component of the bandwidth (i.e., filter_up = [100,130])
- redshift_array: this array is used to test the quality of the redshift search.
- includeCI: Default is False, but if you are certain the observations are deep enough to detect atomic carbon. 
- nr_of_CO_lines: the number of CO-transitions from CO(1-0) up to CO(nr_of_CO_lines,nr_of_CO_lines - 1) that are considered in the redshift determination
- lin_arr_size: to deal with the massive data sizes of (smoothed) redshift samples, a linear redshift space is created to measure the redshift search quality. The redshift_array then samples this linear redshift space using nearest-neighbour interpolation to assess the total redshift search quality.
- sigma_threshold: The minimum distance in signal-to-noise ratio between the potential redshift solutions to allow the use of photometric redshifts as a robust redshift identifier.
- dzUncertainty: The uncertainty in dz used for the signal-to-noise distance between potential redshift solutions.


































