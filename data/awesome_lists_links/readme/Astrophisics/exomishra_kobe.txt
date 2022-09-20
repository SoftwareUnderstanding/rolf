# KOBE - Kepler Observes Bern Exoplanets

![](/images/kobe_compressed.png)

## What are Exoplanets? 

Exoplanets, or extra-solar planets, are planets which orbits stars other than our Sun. Exoplanets are discovered by several detection methods ([see this NASA website](https://exoplanets.nasa.gov/alien-worlds/ways-to-find-a-planet/#) for clear explanations and cool animations). 

## What is the Transit Method?

One of the exoplanet detection methods is called the Transit Method. When a planet passes in front of a star, it blocks some of the [star-light](https://www.youtube.com/watch?v=Pgum6OT_VH8). NASA's space telescope, [*the Kepler Mission*](https://www.nasa.gov/mission_pages/kepler/overview/index.html), discovered several thousands of exoplanets using the transit method.  

However, the transit method has three limitations:
* **Geometrical Limitation**: The transit method can only find those exoplanets, whose orbits are aligned from our point-of-view. Technically, the sky-projected inclination of an exoplanet's orbit needs to be close to 90&deg;. 
* **Detection Biases**: Physically, large planets closely orbitting a small quiet star produce a *much* stronger transit signal-to-noise ratio. This means that planets which are easy to find are found more often, while hard-to-find planets remain elusive. 
* **Completeness and Reliability**: What if any other object, besides a planet, were to periodically transit a star? They may also generate a transit signal. These are called False Positives. We can try to understand these signals and try to get rid of them. But, what if we mis-identify a real exoplanet signal as a false positive? These are called False Negatives. Understanding both false positives and false negatives, allows us a better understanding of true positives and true negatives. This tells us how reliable and complete are the findings of any survey, in general. 

## Why do we need KOBE?

Essentially, we want to have a deeper and better understanding of the cosmos and its intricate physical inter-connectedness. To this end, we need to compare our theoretical understanding of any physical phenomenon with nature via experiments or observations. 

**For Exoplanets:** In order to understand how planets are formed, we can simulate the environment in which they are born, namely - protoplanetary disks (check out these deeply moving and fabulous [images of protoplanetar disks from ALMA](https://www.almaobservatory.org/en/images/almas-high-resolution-images-of-nearby-protoplanetary-disks/)). Using theoretical and numerical calculations we can simulate and study the growth of several thousands of planets. 

KOBE is a program which allows theoretically simulated planets to be compared with exoplanets found by the transit method. This allows us to study so-many things, statistically!

## How does KOBE work?

KOBE adds the geometrical limitations and the physical detection biases of the transit method (mentioned above) to a given populations of theoretical planets. In addition, it also adds the completeness and reliability of a transit survey.

KOBE has three modules for this: KOBE Shadows, KOBE Transits, and KOBE vetter. Each module takes care of each limitation mentioned above. To read more about the details, see  my paper [Mishra et. al. 2021](https://ui.adsabs.harvard.edu/abs/2021arXiv210512745M).

Here are a few images of synthetic systems as their transit geometry is examined by KOBE. The colored bands shows the Transit Shadow Band for all planets in the system.  
<img src="/images/ng76_sys854.png" width="250" height="250"/>
<img src="/images/ng76_sys269.png" width="250" height="250"/>
<img src="/images/ng74_sys661.png" width="250" height="250"/>

## What can I use KOBE for?

You can use KOBE for your studies! 

If you want to compare a theoretical population of planets with exoplanets found by a transit survey, you can use KOBE to bias your theoretical population. 

## Where does KOBE's name come from?

KOBE stands for Kepler Observes Bern Exoplanets. The name reflects its origin. The first project where KOBE was used had:
* Transit Survey - Kepler space telescope
* Theoretical planet formation model - Bern Model

The way I imagined it was: that somehow the telescope Kepler was pointing at Earth, in my computer, and looking at the planetary systems formed by the Bern Model. Literally, Kepler Observes Bern Exoplanets.

## Can KOBE be used for future missions? 

Yes! 
KOBE can be, **in-principle** used for other transit missions/surveys like: [PLATO](https://sci.esa.int/web/plato/), [TESS](https://tess.mit.edu/), etc.

## Publications:

List of publications utilizing KOBE:

1. [Mishra et. al. 2021](https://ui.adsabs.harvard.edu/abs/2021arXiv210512745M/abstract)  
Original publication introducing KOBE. Analyze and comparing the architecture of theoretical planetary systems formed by the Bern Model, with the exoplanetary systems found by Kepler. [Read more about peas in a pod](https://ui.adsabs.harvard.edu/abs/2021arXiv210512745M/abstract)

2. Mishra et. al. 2021 (in prep.)  
Statistically comparing planetary populations from the Bern Model and exoplanets found by Kepler.

## Contact me

For any more details or comments or suggesitons, you can contact me via [my website](https://www.lokeshmishra.com/). 
