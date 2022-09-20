## Stacked exomoons with the Orbital Sampling Effect 
*If you make use of this code, please cite this repository in combination with our [paper](http://iopscience.iop.org/article/10.3847/0004-637X/820/2/88/meta)  
MODELING THE ORBITAL SAMPLING EFFECT OF EXTRASOLAR MOONS  
René Heller, Michael Hippke, and Brian Jackson  
The Astrophysical Journal, Volume 820, Number 2*  


How to get started:

1. Define parameters for the star, planet and moon [all as floats]:
  * Star: Stellar radius, limb-darkening parameters
  * Planet: Radius, semimajor axis, impact parameter, period
  * Moon: Radius, semimajor axis, eccentricity, ascending node, longitude at periastron, inclination

2. Define simulation parameters:
  *  Account for mutual planet/moon eclipses [bool]
  *  Show the planet in the plots [bool]
  *  Add noise to the simulation [ppm/minute, can be zero]
  *  How many transits are to be sampled [integer]
  *  How many (randomly chosen) transits are observed [integer, 0=all]
  *  Moon phase to highlight in plots [0..1]
  *  Quality setting [integer, defines pixel radius of star; scales other bodies]
   
#### Generate a 3D-view of these parameters for visual verification:
 
![ScreenShot](http://www.jaekle.info/osescreenshots/git1.png)

[Klick here for an animation at mid-transit.](http://jaekle.info/osescreenshots/osegif.gif)

As this plot is for illustration purpose only, we visualize the Sun-Earth-Moon system but we scale the physical radii of Earth and the Moon by a factor of ten to make their disks visible against the solar disk. The impact parameters of the planet (b=0.4) and the moon's Keplerian elements (e=0.7, i=83°, Omega=0.7) are chosen arbitrarily for good visual clarity. The plot shows the situation at planetary mid-transit.

In our numerical implementation, the planet-moon ensemble transits the star from left to right. The motion of the planet and the moon around their common barycenter during transit can be included, or neglected, in our code. For most purposes, it is useful to neglect this motion, as the converged OSE curve will not be affected as the motion of both the planet and the moon would be smeared out and averaged in a phase-folded light curve anyways.

#### Generate a riverplot covering phase 0..1:

![ScreenShot](http://www.jaekle.info/osescreenshots/git2.png)

This river plot shows the current moon's position (triangle), as in the figure above. Times of mutual planet/moon transits around phase=0 and phase=0.8 cause less flux loss due to the moon, and are shown in white color. 

#### Generate the stacked lightcurve:

![ScreenShot](http://www.jaekle.info/osescreenshots/git3.png)

Using PyOSE, we simulate one planetary transit and a range of moon transits for various orbital positions of the moon using the transit model of Mandel & Agol (2002). The planetary transit and the averaged moon transit light curve are then combined into one pseudo-phase-folded, average light curve that contains the OSE. For the moon transits, we sample n transits equally spaced in time. Of course, for eccentric moon orbits, the satellite's orbital speed varies, so that when sampling time in equal steps, the satellite's positional change is also variable. Our code fully accounts for this.

#### Calculate the total occulted stellar flux: 
```
-7916.17381004 ppm hrs
```

You can save all figures as PDF, PNG, EPS, etc., and the time/flux data as CSV, Excel etc.
