# beamModelTester

beamModelTester is a general-purpose tool that enables evaluation of models 
of the variation in sensitivity and apparent polarisation of fixed antenna phased array 
radio telescopes.  

The sensitivity of such instruments varies with respect to the orientation
of the source to the antenna.  This creates a variation in sensitivity over altitude and azimuth.
Further geometric effects mean that this variation is not conisistent with respect to frequency.
In addition, the different relative orientation of orthogonal pairs of linear antennae produces 
a difference in sensitivity between the antennae, leading to an artificial apparent polarisation

By comparing the model with observations made using the given telescope, it is possible to
evaluate the model's performance.  The results of this evaluation can be used to provide a 
figure of merit for the model, and also to guide improvements to the model.  

As an additional feature, this system enables plotting of results from a single station observation on a variety of parameters.

BeamModelTester was developed as part of the RadioNet RINGS (Radio Interferometry Next Generation Software) JRA. RadioNet has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 730562.


## Table of Contents<a name="ToC"></a>
1. [System Requirements](#sys_req)
    1.  [Language and Libraries](#languages)
    1.  [Depenent packages](#dependencies)
    1.  [Operating System](#os)
1.  [Installation](#install)
1.  [How to use](#howto)
    1.  [Tutorials](#tutorial)
    1.  [Plotting software](#plotting)
    1.  [Full Pipeline Data Processing](#pipeline)
1.  [System Design Components](#design)

## System Requirements<a name="sys_req"></a>

### Language and Libraries<a name="languages"></a>
This software runs in **Python 2.7**.  Ensure the following Libraries are installed and up-to-date
in your python environment.  

To install, run pip install \<package\>\
To update, run pip update \<package\>

 * Required: pandas, numpy, sys, argparse, os, h5py, matplotlib, scipy
 * Recommended: astropy (Horizontal coordinate plotting will not work without this package)

### Dependent packages<a name="dependencies"></a>
The following 3rd-party packages (not available as pip packages) are required for full functionality of this system. 
Some functionality may work without these packages, but installation is recommended.  
Follow the links below to install these packges
  * [dreamBeam](https://github.com/2baOrNot2ba/dreamBeam)
  * [iLiSA](https://github.com/2baOrNot2ba/iLiSA)
  * [python-casacore](https://github.com/casacore/python-casacore)

### Operating System<a name="os"></a>
* Recommended OS: Ubuntu 18.04
* Partial functionality available in Windows 8, 10
* Other operating systems not tested, but may work with appropriate Python interpreter.


## Installation<a name="install"></a>
Once the above requirements are met, clone this repository to your local file system.

git clone https://github.com/creaneroDIAS/beamModelTester

## How to use<a name="howto"></a>

### Tutorials<a name="tutorial"></a>
A series of tutorials have been developed to enable a user to learn how to use the system

[Tutorial 1: Basic Plots](/tutorial_1.md) \
[Tutorial 2: Plotting Options](/tutorial_2.md) \
[Tutorial 3: Frequency Options](/tutorial_3.md) \
[Tutorial 4: File I/O](/tutorial_4.md) \
[Tutorial 5: Location and Target](/tutorial_5.md) \
[Tutorial 6: Figures of Merit](/tutorial_6.md) \
[Tutorial 7: Offsets and Scaling](/tutorial_7.md) \
[Tutorial 8: Visual Options](/tutorial_8.md) 

### Plotting software<a name="plotting"></a>
This software operates in three modes: GUI interactive, Command Line interactive and Non-interactive mode.


GUI Interactive mode can be started by running the **rungui.sh** script and following [these instructions](/comparison_module/interactive_mode.md)

Command Line Interactive mode can be started by running the **runcli.sh** script and following [these instructions](/comparison_module/interactive_mode.md)

Non-interactive mode is **recommended for advanced users only** especially users working via SSH with no DISPLAY variable set.  
To run non-interactively, call the [comparison script] directly and use the arguments discussed in 
[this document](/comparison_module/cli_arguments.md)

e.g. ***./comparison_module/comparison_module_1_0.py --model ~/SE607_24h_sim.csv --scope ~/SE607_2018-03-16T15_58_25_rcu5_CasA_dur86146_ct20161220_acc2bst.hdf5 --values xx yy --plots spectra model scope diff -I 0***

### Full Pipeline Data Processing<a name="pipeline"></a>
Acquire ACC Data from LOFAR and store it in a directory with the following name structure

*{STN_ID}_YYYYMMDD_HHMMSS_rcu{RCU_MODE}_dur{DURATION}_{SOURCE}_acc*\
e.g. *IE613_20180406_091321_rcu3_dur91863_CasA_acc*

A sample of suitable data is available at https://zenodo.org/record/1326532#.W3L8FNVKiUk

Run the [data extraction script](https://github.com/creaneroDIAS/beamWrapper/blob/master/data_wrapper.sh) 
with that directory as an argument.\
e.g ***./beamWrapper/data_wrapper.sh ~/IE613_20180406_091321_rcu3_dur85628_CasA_acc***

This will produce a [HDF5 file](/data_descriptions/OSO_HDF5.md)
and a [CSV file](/data_descriptions/DreamBeam_Source_data_description.md) which can be used in the next step
or otherwise as needed.

## System Design Components<a name="design"></a>

There are three major components to this system:
  * Data from the Telescope (Currently LOFAR [ACC files](/data_descriptions/ACC_Source_data_description_0_0.md) converted to [HDF5](/data_descriptions/OSO_HDF5.md) by [iLiSA](https://github.com/2baOrNot2ba/iLiSA))
  * Data from the Model (Currently [CSV Files](/data_descriptions/DreamBeam_Source_data_description.md) based on the Hamaker model as output from [dreamBeam](https://github.com/2baOrNot2ba/dreamBeam))
  * Comparison/Analysis [module design at this link](/comparison_module/readme.md)
  
Software Design Documents are available at [This Link](/overall_design.md)

![Design Diagram](images/testHarness_Fig1v3.PNG)
  
Extraction of data, especially observed data, can be time-consuming.  As a result, separate scripts are provided to 
[extract the data](https://github.com/creaneroDIAS/beamWrapper/blob/master/data_wrapper.sh) 
and to [analyse it](/comparison_module/comparison_module_1_0.py).
An [overall script](https://github.com/creaneroDIAS/beamWrapper/blob/master/complete_wrapper.sh) 
which calls all three components of the software is provided, but usually the data extraction routines are carried out once, 
but the analysis and visualisations are repeated, so the use of this script is deprecated. *Currently a minor bug in this to be worked out*






