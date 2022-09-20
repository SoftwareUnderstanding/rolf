# TransitSOM
A Self-Organising Map for Kepler and K2 Transits


Requirements:

-Python 2.7
-NumPy
-PYMVPA (http://www.pymvpa.org/)

User-facing functions are in TransitSOM_release.py with descriptions. Other libraries contain utility functions.

Key Functions are:

TransitSOM_release.PrepareLightcurves() - Takes a list of files and creates an array of binned transits suitable for classification or training a new SOM.

TransitSOM_release.PrepareOneLightcurve() - Takes a single lightcurve and creates an array of bins suitable for classification.

TransitSOM_release.CreateSOM() - Trains a new SOM using the output of PrepareLightcurves(). Can save to disk.

TransitSOM_release.ClassifyPlanet() - Produces Planet/False positive statistics from Armstrong et al 2016 using the output of PrepareLightcurves() or PrepareOneLightcurve. Can use User-defined SOM or the SOM from Armstrong et al 2016.

TransitSOM_release.LoadSOM() - loads a SOM that has been saved using CreateSOM()

***TESTING IS ONGOING, PLEASE CONTACT d.j.armstrong@warwick.ac.uk if problems arise***

