extinction-distances
====================

Code to implement the blue star number count method from Foster et al. (2012) http://adsabs.harvard.edu/abs/2012ApJ...751..157F

This code requires a modified version of the astroquery package. You will need 
to download and install my fork of the astroquery package from 
https://github.com/jfoster17/astroquery. This branch has two main differences 
from the official branch -- (1) it adds a VISTA module which has the same interface 
as the UKIDSS module (2) it updates the default columns returned from a 
Besancon query, since I could not get that working without changing the defaults.

All the work is done inside the BaseDistObj. An example run for a single cloud 
would be as follows:

Import the package

    from extinction_distance.distance import BaseDistObj
    
Create a distance/cloud object, specifying which surveys to use. Typically 
this is nir\_survey="UKIDSS" in the north and "VISTA" in the south and 
cont\_survey="BGPS" in the north and "ATLASGAL" in the south. Files are 
all stored in a directory with the name you give as name. Position 
must be given in Galactic coordinates.

    cloud = BaseDistObj.BaseDistObj(name,(glon,glat), 
              nir_survey="VISTA",cont_survey="ATLASGAL")
              
Fetch the data and model. These steps normally check to see if the data 
exists before re-fetching anything. Use argument clobber=True to force 
the program to refetch this data.

    cloud.get_nir_images()
    cloud.get_nir_cat()
    cloud.get_continuum()
    cloud.get_model()

Identify the cloud contours from the continuum image. This step 
is quick and is NOT CACHED. Thus, it needs to be run every time.

    cloud.get_contours(cloud.continuum)

Make an image showing the three-color NIR image and the identified 
contour from the continuum image.

    cloud.show_contours_on_threecolor()

Do photometry and estimate completeness for the NIR image. By default 
the completeness step (which is very slow) will only run the first time 
for each object. Use force\_completeness = True to force regenerating 
the completeness estimate.

    cloud.make_photo_catalog()

Generate the distance estimate. This is printed out to the 
terminal and saved as a plot showing the confidence interval. The 
diagnostic plots should be checked before blindly believing any 
distance estimate.

    cloud.do_distance_estimate()