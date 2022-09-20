encube
========
Large-scale comparative visualisation and analysis of sets of multidimensional data

Copyright (c) 2015, Dany Vohl, David G. Barnes, Christopher J. Fluke,
Govinda Poudel, Nellie Georgiou-Karistianis, Amr H. Hassan, Yuri Benovitski,
Tsz Ho Wong, Owen Kaluza, Toan D. Nguyen, C. Paul Bonnington. All rights reserved.

Authors: Dany Vohl - dvohl (at) swin.edu.au, David G. Barnes - david.g.barnes (at) monash.edu,
Chris J. Fluke - cfluke (at) swin.edu.au, Yuri Benovitski - ybenovitski (at) bionicsinstitute.org,
Owen Kaluza - owen.kaluza ( at ) monash.edu, Toan D. Nguyen toan.nguyen (at) monash.edu.

encube is licensed under the GNU General Public License 
(version 3) as published by the Free Software Foundation.
Find more at [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).

Dependencies:
-------------
S2PLOT : [https://github.com/mivp/s2plot](https://github.com/mivp/s2plot)

S2VOLSURF: [https://github.com/mivp/s2volsurf](https://github.com/mivp/s2volsurf)

CFITSIO (for astronomical data) : [http://heasarc.gsfc.nasa.gov/fitsio/fitsio.html](http://heasarc.gsfc.nasa.gov/fitsio/fitsio.html)

Configuration
-------------
See build setup examples in the scripts repository: typical S2PLOT builds.
See configuration files example in the config repository.

e.g. "app_type": "astro_fits" or "brain_xrw"

How to use it:
--------------
build s2hd (encube-PR): scripts/build-VERSION.

**start encube_PR via encube_M**
python encube_m.py config/config_file.json 0

**start interactive unit server (webserver), which will create an instance of encube_M and connect to encube_PR**
python webserver.py config/config_file.json

**Open your browser and go to http://localhost:8000**

Note:
-----
Tractography shader for IMAGE-HD currently supported on Linux only.

Reference note:
-------------
We would appreciate it if research outcomes using encube would
provide the following acknowledgement:

"Visual analytics of multidimensional data was conducted with encube."

and a reference to

Dany Vohl, David G. Barnes, Christopher J. Fluke, Govinda Poudel, Nellie Georgiou-Karistianis,
Amr H. Hassan, Yuri Benovitski, Tsz Ho Wong, Owen L Kaluza, Toan D. Nguyen, C. Paul Bonnington. (2016) Large-scale 
comparative visualisation of sets of multidimensional data. *PeerJ Computer Science* 2:e88 
https://doi.org/10.7717/peerj-cs.88

Acknowlegements:
----------------
This work was enabled and supported by the Monash Immersive Visualisation Platform [http://monash.edu/mivp](http://monash.edu/mivp).
