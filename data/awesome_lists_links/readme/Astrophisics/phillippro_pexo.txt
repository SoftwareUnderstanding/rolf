# PEXO v2.0
[![DOI](https://zenodo.org/badge/210655784.svg)](https://zenodo.org/badge/latestdoi/210655784)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Compared with previous models and packages, PEXO is general enough to account for binary motion and stellar reflex motions induced by planetary companions. PEXO is precise enough to treat various relativistic effects both in the Solar System and in the target system (Roemer, Shapiro, and Einstein delays).

PEXO is able to model timing to a precision of 1 ns, astrometry to a precision of 1 μas, and radial velocity to a precision of 1 cm/s. There are [pdf](https://github.com/phillippro/pexo/blob/master/docs/manual.pdf) and [html](http://rpubs.com/Fabo/pexo2) versions of the manual available for instructions of how to use PEXO. The fitting mode and a python wrapper are in development and expected to be released soon.

The relevant paper was published by ApJS. If you use PEXO in your work, please cite the paper:
```
@article{Feng_2019,
	doi = {10.3847/1538-4365/ab40b6},
	url = {https://doi.org/10.3847%2F1538-4365%2Fab40b6},
	year = 2019,
	month = {oct},
	publisher = {American Astronomical Society},
	volume = {244},
	number = {2},
	pages = {39},
	author = {Fabo Feng and Maksym Lisogorskyi and Hugh R. A. Jones and Sergei M. Kopeikin and R. Paul Butler and Guillem Anglada-Escud{\'{e}} and Alan P. Boss},
	title = {{PEXO}: A Global Modeling Framework for Nanosecond Timing, Microarcsecond Astrometry, and $\upmu$m s-1 Radial Velocities},
	journal = {The Astrophysical Journal Supplement Series}
}
```


## Binder

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phillippro/pexo/HEAD?filepath=demos%2FIntroduction.ipynb) <sup>[What is Binder?](https://mybinder.readthedocs.io/en/latest/)</sup>

Binder is a free open-source tool that creates custom computing environments that can be shared and used by many remote users. 
It's the easiest way to try the code without the hassle of installation.
You can run PEXO in your browser by following the [binder link](https://mybinder.org/v2/gh/phillippro/pexo/HEAD?filepath=demos%2FIntroduction.ipynb). Note that it may take up to 10 minutes to set up the environment for PEXO depending on the server load.


## Local installation

You will need [`conda`](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to install the dependencies. This is the easiest way, but you can install the dependencies manually if you prefer, they are listed in [`environment.yml`](environment.yml).

1. Clone this repository:

```bash
git clone https://github.com/phillippro/pexo.git
```

2. [Install](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-from-file) the `conda` environment from `environment.yml`. This step might take a few minutes.

```bash
cd pexo
conda env create -f environment.yml
```

3. Activate the environment (named `pexo` by default). You will need to do this for any new terminal session.

```bash
conda activate pexo
```

4. PEXO is ready to run! Refer to the [documentation](http://rpubs.com/Fabo/pexo2) and [demos](demos/Introduction.ipynb) for guidance.

### Demos

To run the demos you will need [JupyterLab](https://jupyter.org/install) installed and make it use PEXO conda environment by running `python -m ipykernel install --user --name=pexo`.
