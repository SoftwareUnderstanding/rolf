# Hilal-Obs
Authentication Algorithm for New Moon Visibility Report

Hilal Obs is a method to authenticate contestable lunar crescent visibility records. Hilal Obs utilized Schaefer Model of Lunar Crescent Visibility and Crumey Model Contrast Threshold.

The Coding require Skyfield to works, and works best within the Anaconda Environment. So it is advisable to download Anaconda and Skyfield Library.

Example of the data is described in Input File. The Input file must be in csv format, and require, Latitude (Lat), Longitude (Long), Day, 
Month, Year, Level of Light Pollution (LP) and Time Zone (TZ).

The output will result in "Modified Schaefer Prediction Model" and "Crumey Prediction Model" column on the first row. On the next row, is the result of the Method, a visible prediction will result as "Moon Sighted", while an invisible moon sighting prediction will result as empty Non Value result. Thhe example of the output is described as Output File.

Makesure to edit the input and output Path file under #Input Path and #Output Path.
