# Umbrella2

Umbrella2 is a highly modular, open-source library for asteroid detection written under dotnet.

Functions are provided for:
* Reading and writing FITS files
* Badpixel removal
* Noise removal
* Object detection, including very faint trails
* Importing detections from Source Extractor
* Fixed star removal (from images and/or detection lists)
* Object and tracklet filtering
* Object pairing
* Object and tracklet display
* Reading and writing MPC optical reports
* SkyBoT and VizieR querying

In addition, user image processing functions can take advantage of the algorithms framework and use the multi-threaded CPU scheduler for easy algorithm parallelization.

### Note

The master branch follows development (currently v3) and the API is unstable. Please use stable branches if you need to compile the library.
