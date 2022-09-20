# minot: Modeling the ICM (Non-)thermal content and Observable prediction Tools
Software dedicated to provide a self-consistent modeling framework for the thermal and the non-thermal diffuse components in galaxy clusters, and provide multi-wavelenght observables predictions.
                                                            
## Overview of the physical processes and structure of the code
<figure>
	<img src="/overview1.png" width="600" />
	<figcaption> Figure 1. Overview of the parametrization, physical processes, and observables dependencies.</figcaption>
</figure>

<p style="margin-bottom:3cm;"> </p>

<figure>
	<img src="/overview2.png" width="600" />
	<figcaption> Figure 2. The structure of the code. </figcaption>
</figure>

## Content
The minot directory contains the main code, including:

- model.py : 
	main code that defines the class Cluster
    
- model_admin.py : 
        subclass that defines administrative tools
   
- model_modpar.py : 
        subclass that handles model parameters functions 
        
- model_phys.py : 
    subclass that handles the physical properties of the cluster
    
- model_obs.py : 
    subclass that handles the observational properties of the cluster
    
- model_plots.py : 
        plotting tools for automatic outputs

- model_title.py : 
	title for the package

- ClusterTools :
    Repository that gather several useful libraries

The root directory also provides a set of examples:

- notebook :
	Repository where to find Jupyter notebook used for validation/example. 

## Environment
To be compliant with other softwares developed in parallel, the code was originally developed in python 2. Recently, the code was made compatible with python 3.

## Installation
You can use pip to install the package:

```
pip install minot
```

#### Dependencies 
The software depends on standard python packages:
- astropy
- numpy
- scipy
- matplotlib

But also:
- ebltable (see https://github.com/me-manu/ebltable)
- healpy (optional, see https://healpy.readthedocs.io/en/latest). Healpy is not automatically installed, but you can do it independently to benefit of a view functions that use healpy, but are not directly used in minot.

In the case of X-ray outputs, it will be necessary to have the XSPEC software installed independently (https://heasarc.gsfc.nasa.gov/xanadu/xspec/).

#### Encountered issues
- Depending on the python version, the automatic installation of healpy does not work. As healpy is optional, it was removed from the dependencies and healpy can be installed independently if necessary.

- For MAC-OS, in some version of python 2, the automatic installation of matplotlib may lead to an error related to the backend when importing matplotlib.pyplot. In this case, reinstalling matplotlib using conda, as `conda install matplotlib` should solve the problem.

- The automatic installation of dependencies is sometimes misbehaving. In such case, you may just install the required packages independently:

`conda install astropy`

`conda install numpy`

`conda install scipy`

`conda install matplotlib`

#### Reference
In case you use minot in your research, you can cite R. Adam, H. Goksu, A. Leingärtner-Goth, et al. (2020) to acknowledge its use. The paper is availlable here and contains the full description of the code: https://ui.adsabs.harvard.edu/abs/2020arXiv200905373A/abstract. This also https://www.aanda.org/articles/aa/full_html/2020/12/aa39091-20/aa39091-20.html

#### History
- Version 0.1.0 --> Initial release

- Version 0.1.1 --> Correction of warnings and minor bugs

