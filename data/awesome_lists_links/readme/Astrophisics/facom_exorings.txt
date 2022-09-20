Exorings
========

Code Repository for Exoring Transit Calculations
------------------------------------------------

Quick start
-----------

- Download the package:

  Using git:

  ```
  $ git clone http://github.com/facom/exorings.git
  ```

  Or download it as a zip file from [this
  url](https://github.com/facom/exorings/archive/master.zip).

- Run a test:

  ```
  $ python exorings.py
  ```
	
  This test will tell you which python packages are required to use
  the **exorings** code.

- Calculate basic ring transit properties:

  ```
  $ python exorings-basic.py fi=1.5 fe=2.35 theta=30.0 ir=80.0
  ```

  Output include: transit depth (in ppm), total transit duration (in
  hours), duration of full transit (in hours), observed radius (pobs),
  observed asterodensity (rhoobs).

References
----------

When using **exorings** code for research purposes please cite the
following paper:
   
> *Zuluaga, J.I., Kipping, D., Sucerquia, M., Alvarado, J. A.* "**A
> novel method for identifying exoplanetary rings**", Submitted to
> Astrophysical Journal Letters, 2015.

==================================================
Jorge I. Zuluaga (C) 2015
