# GalLenspy

Gallenspy is an open source code created in python, designed for the mass profiles reconstruction in
disc-like galaxies using the GLE. It is important to note, that this algorithm allow to invert numerically
the lens equation for gravitational potentials with spherical symmetry, in addition to the estimation in
the position of the source , given the positions of the images produced by the lens. Also it is important
to note others tasks of Gallenspy as compute of critical and caustic curves and obtention of the Einstein
ring.
The main libraries used in Gallenspy are: **numpy** for the data hadling, **matplotlib** regarding the
generation of graphic interfaces, **galpy** to obtain mass superficial densities, as to the parametric adjust
with Markov-Montecarlo chains is taken into account **emcee** and for the graphics of reliability regions
**corner** is used.

## How to use Gallenspy

To start Gallenspy, it is important to give the values of cosmological distances in Kpc and critical
density in SolarMass/Kpc² units for the critical density, which are introduced by means of a file named
**Cosmological_distances.txt**. On the other hand, it is the **coordinates.txt** file where the user must
introduced the coordinates of the observational images and its errors respetively (in radians).(Note: for
the case of a circular source is present the **alpha.txt** file, where the user must introduced angles value
in radians belonging to each point of the observational images). These files mut be in each folder of
Gallenspy which execute distinct tasks.

### Source estimation

In the case of the estimation of the source, Gallenspy let to the user made a visual fitting in the notebook
**Interactive_data.ipynb** for a lens model of exponential disc in the folder **Source_estimation**, 
where from this set of estimated parameters the user have the posibility of established the initial guess.

How **Interactive_data.ipynb** is an open source code, the user has the possibility of modify the
parametric range in the follow block of the notebook.

![imagen2](https://user-images.githubusercontent.com/32373393/119746450-23adf500-be56-11eb-8cd4-04c1c0766c50.png)

With the values of visual fitting, the user must click on the **Finish** button which generates the file named
**init_guess_params.txt**. 

![imagen1](https://user-images.githubusercontent.com/32373393/119743961-974d0380-be50-11eb-88ad-bd3bff9fc208.png)

With the **init_guess_params.txt** file generated, the user can execute the  **source_lens.py** file in the 
following way:

**$ python3 source_lens.py**

Finally Gallenspy generate a files set denominated **parameters_lens_source.txt**, **contours_source_lens.pdf** and 
**fitting.pdf**.

![imagen3](https://user-images.githubusercontent.com/32373393/119750488-0e899400-be5f-11eb-82ef-5700d9d0d0ba.png)

![imagen5](https://user-images.githubusercontent.com/32373393/119751268-6ffe3280-be60-11eb-9d9c-88881977996d.png)

The file **parameters_lens_source.txt** must be copy in Einstein_Ring and Mass_reconstruction folders .

### Mass reconstruction

In the case of mass reconstruction, Gallenspy let to the user made a visual fitting in the notebook **Interactive_data.ipynb** 
for a lens model with exponential disc, NFW, burket and Miyamoto Nagai profiles in the folder **Mass_reconstruction**, where from 
this set of estimated parameters the user have the posibility of established the initial guess.

![imagen6](https://user-images.githubusercontent.com/32373393/119752067-ec454580-be61-11eb-8c23-4f26da843af3.png)

How **Interactive_data.ipynb** is an open source code, the user has the possibility of modify the parametric range in the follow block
of the notebook.

![imagen7](https://user-images.githubusercontent.com/32373393/119752638-fca9f000-be62-11eb-8413-2230197c8b71.png)

With the values of the visual fitting, and the **init_guess_params.txt** file the user must execute the file **parameters_estimation.py**
in the following way:

**$ python3 parameters_estimation.py**

Finally Gallenspy generate a files set denominated **final_params.txt**, **contours.pdf** and **fitting.pdf**.

![imagen9](https://user-images.githubusercontent.com/32373393/119753378-4cd58200-be64-11eb-97ac-8a34ef2f3d49.png)

### Einstein Ring

For this task, the user must execute the file Einstein_ring.py in the folder Einstein_ring, in the following way:

**$ python3 Einstein_ring.py**

Finally Gallenspy generate a file denominated **Einstein_radius.txt**, with the value of the Einstein radius and its 
errors respectively.

Other file generate in Gallenspy is **Einstein_radius_AND_images.pdf**.

![imagen10](https://user-images.githubusercontent.com/32373393/119754039-73e08380-be65-11eb-9da5-833ddc8dbf69.png)

**The related pre-print reference: Mass reconstruction in disc like galaxiesusing strong lensing and rotation curves: The Gallenspy package.**
