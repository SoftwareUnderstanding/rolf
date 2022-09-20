# JWST_Simulation
Three novel python files are included which simulate the JWST deep field image
given the characteristics of the NIRCam. The deterministic simulation generates
a deep field image from one set of initial conditions defined in the code's 
control panel. The second code runs the deterministic simulation as an ensemble
and outputs the average galaxy coverage percentage from averaging the ensemble 
members. The third code generates a negative of the deep field image as a more 
realistic representation of what the JWST deep field image will look like, but
requires an open source negative to positive converter. Comments are included
throughout the code to describe what each line of code is doing. The results of 
the study conducted with these codes will be submitted for publication. All three
codes must be run through google collaboratory or issues may arise with some of
the processes. Any NIRCam filter can be selected in the control panel.
