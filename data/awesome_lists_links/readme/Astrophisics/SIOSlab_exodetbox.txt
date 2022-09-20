# exodetbox
<img src="https://raw.githubusercontent.com/SIOSlab/exodetbox/main/documentation/logo/exo-det-boxlogo.svg" width="150" height="150" />
Methods for finding planets sharing a common (s,dmag)

[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)

Keithly D., Savransky D., Spohn C., Integration Time Adjusted Completeness, J. of Astronomical Telescopes, Instruments, and Systems, 7(3), 037002 (2021). https://doi.org/10.1117/1.JATIS.7.3.037002


## For Beginners
For a new person to exodetbox but experienced with EXOSIMS (or interested in seeing how exodetbox can be applied), the first thing to try is testing_EXOSIMS_IAC.py.
testing_EXOSIMS_IAC.py instantiates an EXOSIMS Mission Sim object which uses the EXOSIMS "Integration Time Adjusted Completeness" module for computing completeness when comp_calc is called. The EXOSIMS IAC module simply imports "from exodetbox.projectedEllipse import \*" and calls "integrationTimeAdjustedCompletness".

An example dynamic completeness calculation can be seen in "dynamicCompleteness.py". This example goes from simulated planets to computing true anomaly visibility windows, converting these into times pas periastron and finally invoking the "dynamicCompleteness" function.

## Brown2010DynamicCompletenessReplication.py
Plots Dynamic Completeness using Brown 2010's method as implemented by Corey Spohn using both the Lambert phase function and Quasi-Lambert phase function
Runs Dynamic Completeness (a method in this script) using the exodetbox methods in projectedEllipse and plots the revisit completeness and dynamic completeness.

## corey_dynamic_completeness.py
Calculates Dynamic Completeness using Corey's Method

## dynamicCompleteness.py
Calculates and Plots Dynamic Completeness for multiple subtypes of exoplanets

## keithlyCompletenessConvergence.py
Calculates convergence of completeness calculated using exodetbox projectedEllipse methods

## plotProjectedEllipse.py
Contains plotting functions called by testing_projectedEllipse.py.
Meant to contain plotting utilities for intermediate steps of and final plots for projectedEllipse.py

## projectedEllipse.py
The workhorse script.
Contains all functions necessary to calculate a planets s,dmag extrema as well as when a planet intersects a given s or dmag (if it does intersect them).
also projects the 3D Keplerian Orbit into a reparameterized 2D ellipse in the plane of the sky.

## testing_EXOSIMS_IAC.py
A simple script demonstrating how to use integration time adjusted completeness in EXOSIMS (slow implementation).

## testing_projectedEllipse.py
Generates plots used in Keithly 2021 like integration time adjusted completeness vs integration time for planets at verying distances.
Additionally plots intermediate steps like the ellipse derotation, plane of the sky apparent circle ellipse intersections, plots separation and dmag vs nu and time

## trueAnomalyFromEccentricAnomaly.py 
A simple function for calculating the true anomaly from the eccentric anomaly
