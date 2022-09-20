```
################################################################################
  ____  _    _ __  __           _      
 |  _ \| |  | |  \/  |         | |     
 | |_) | |__| | \  / | ___ __ _| | ___ 
 |  _ <|  __  | |\/| |/ __/ _` | |/ __|
 | |_) | |  | | |  | | (_| (_| | | (__ 
 |____/|_|  |_|_|  |_|\___\__,_|_|\___|

 The Binary Habitability Mechanism Calculator
 2014 [)] Jorge Zuluaga - Viva la BHM!
################################################################################
```

Presentation
------------

**BHMcalc** is a project conceived and developed after the
collaboration of the author (Prof. **Jorge I. Zuluaga, UdeA**) with
Prof. **Paul Mason** (NMSU) and their former graduate and
undergraduate students Dr. Pablo A. Cuartas-Restrepo and Joni Clark.
It arose from the interest of our team (internally dubbed as the *BHM
Mafia*) in the complex and rich conditions that potentially habitable
planets could have while orbiting moderately separated binaries
(binary periods under 60 days).

This is a tool that can be used as a web app or a python package.  The
tool is entirely programmed in python and php (web interface) and it
intends to be as friendly as possible to astronomers working in this
field.

If you need to cite this software please include a reference to one
(or all) of the following papers:

- Mason, P. A., Zuluaga, J. I., Clark, J. M., & Cuartas-Restrepo,
  P. A. (2013). Rotational Synchronization May Enhance Habitability
  for Circumbinary Planets: Kepler Binary Case Studies. The
  Astrophysical Journal Letters, 774(2), L26 ([ADS
  Entry](http://adsabs.harvard.edu/abs/2013ApJ...774L..26M)).

- Mason, P. A., Zuluaga, J. I., Cuartas-Restrepo, P. A., & Clark,
  J. M. (2014). Circumbinary Habitability Niches. arXiv preprint
  arXiv:1408.5163 ([ADS
  Entry](http://adsabs.harvard.edu/abs/2014arXiv1408.5163M))

Versions
--------

An on-line version of BHMcalc is available at:
http://bit.ly/BHM-calculator.  

This version is intended to be used by non-programmer or for those not
interested on dealing with source code coming from a different
programmers.

For those interested in getting the source code or installing a mirror
site, you need to know that presently are two versions of the
software.  The first (and more primitive version) located in the trunk
of this project (master branch).  And the newest (experimental but
more flexible and faster) version which is presently located in the
`BHMcalc2` branch.

For getting the lattest version please follow the instructions below.

The first version is not anymore maintained so it is not as
recommended as the newest version.

Getting a copy
--------------

To get a copy of the newest version of this project just execute:

```
$ git clone --branch BHMcalc2 http://github.com/facom/BHMcalc.git
```

For the oldest version just remove the `--branch` option.

Instructions for the contirbutor
--------------------------------

1. Generate a public key of your account at the client where you will
   develop contributions:
   
   ```
   $ ssh-keygen -t rsa -C "user@email"
   ```

2. Upload public key to the github FACom repository (only authorized
   for the FACom repository manager), https://github.com/facom.

3. Configure git at the client:

   ```
   $ git config --global user.name "Your Name"
   $ git config --global user.email "your@email"
   ```

4. Get an authorized clone of the project:

   ```
   $ git clone git@github.com:facom/BHMcalc.git
   ```

5. Checkout the branch you are interested in (e.g. BHMcalc2):

   ```
   $ git checkout -b BHMcalc2 origin/BHMcalc2
   ```

6. Checkout back into the master:

   ```
   $ git checkout master
   ```

Licensing
---------

This project must be used and distributed under the [GPL License
Version 2] (http://www.gnu.org/licenses/gpl-2.0.html).

All wrongs reserved to [Jorge I. Zuluaga](mailto:zuluagajorge@gmail.com).
