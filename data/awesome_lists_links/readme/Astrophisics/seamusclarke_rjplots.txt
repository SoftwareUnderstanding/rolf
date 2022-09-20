![alt text](https://github.com/SeamusClarke/RJplots/blob/main/Images/RJplots.png)

# RJ-plots: An improved method to classify structures objectively and in an automated manner

RJ-plots is a Python module which uses a moments of inertia method to disentangle a 2D structure's elongation from its centrally over/under-density, thus allowing a means to classify such structures. It may be applied to any 2D pixelated image such as column density maps or moment zero maps of molecular lines. The code is open-source and can be downloaded here.

The accompanying paper will be available shortly. It explains the details of the method, shows the sensitivity of the method to different parametrizable shapes as well as example applications to real astronomical data, both from observations and numerical simulations. 

This work is built upon the J-plots code written by Dr Sarah Jaffa ([found here](https://github.com/SJaffa/Jplots)). 

## Dependencies 

RJplots requires 2 common libraries to be installed:

* NumPy,
* Matplotlib

## Using the code

An example script is provided which shows how to use the package. It calculates the RJ-values for 4 basic shapes and plots the corresponding RJ-plot, as well as provides the classification of the shapes. The exact meanings of the classifications are best described in the accompanying paper. The classifications are given as a number between 1 and 4:

* 1 - Quasi-circular, centrally over-dense,
* 2 - Quasi-circular, centrally under-dense,
* 3 - Elongated, centrally over-dense,
* 4 - Elongated, centrally under-dense.

## Acknowledgements 
This code was produced with the support of the Ministry of Science and Technology (MoST) in Taiwan through grant MoST 108-2112-M-001-004-MY2
## Contact

For up-to-date contact information see my [website](https://seamusclarke.github.io/#five)