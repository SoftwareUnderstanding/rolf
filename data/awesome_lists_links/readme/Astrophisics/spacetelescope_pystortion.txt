[![Build Status](https://travis-ci.org/spacetelescope/pystortion.svg?branch=primary)](https://travis-ci.org/spacetelescope/pystortion)
[![Documentation Status](https://readthedocs.org/projects/pystortion/badge/?version=latest)](https://pystortion.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/spacetelescope/pystortion/badge.svg?branch=primary)](https://coveralls.io/github/spacetelescope/pystortion?branch=master)
[![PyPI version](https://badge.fury.io/py/pystortion.svg)](https://badge.fury.io/py/pystortion)
[![PyPI - License](https://img.shields.io/pypi/l/Django.svg)](https://github.com/spacetelescope/pystortion/blob/primary/licenses/AURA.rst)
[![DOI](https://zenodo.org/badge/157456393.svg)](https://zenodo.org/badge/latestdoi/157456393)


# pystortion
Support for distortion measurements in astronomical imagers.

### Functionalities (work in progress)
* Classes to support fitting of bivariate polynomials of arbitrary degree
* Helper functions for crossmatching catalogs
 
### Installation  
`pip install pystortion`

Or, clone the repository:  
`git clone https://github.com/spacetelescope/pystortion`  
and install pystortion:  
`cd pystortion`  
`python setup.py install` or  
`pip install .`

This package was developed in a python 3.5 environment.   

### Example usage
For crossmatch, please see ``tests/test_crossmatch.py``
   

### Documentation
pystortion is documented at https://pystortion.readthedocs.io/  


### Citation
If you find this package useful, please consider citing the Zenodo record using the DOI badge above.
Please find additional citation instructions in [CITATION](CITATION). 



### Contributing
Please open a new issue or new pull request for bugs, feedback, or new features you would like to see. If there is an issue you would like to work on, please leave a comment and we will be happy to assist. New contributions and contributors are very welcome!   
 Do you have feedback and feature requests? Is there something missing you would like to see? Please open an issue or send an email to the maintainers. This package follows the Spacetelescope [Code of Conduct](CODE_OF_CONDUCT.md) strives to provide a welcoming community to all of our users and contributors. 
 
The following describes the typical work flow for contributing to the pystortion project (adapted from https://github.com/spacetelescope/jwql):
0. Do not commit any sensitive information (e.g. STScI-internal path structures, machine names, user names, passwords, etc.) to this public repository. Git history cannot be erased.
1. Create a fork off of the `spacetelescope` `pystortion` repository on your personal github space.
2. Make a local clone of your fork.
3. Ensure your personal fork is pointing `upstream` to https://github.com/spacetelescope/pystortion
4. Open an issue on `spacetelescope` `pystortion` that describes the need for and nature of the changes you plan to make. This is not necessary for minor changes and fixes. 
5. Create a branch on that personal fork.
6. Make your software changes.
7. Push that branch to your personal GitHub repository, i.e. to `origin`.
8. On the `spacetelescope` `pystortion` repository, create a pull request that merges the branch into `spacetelescope:master`.
9. Assign a reviewer from the team for the pull request.
10. Iterate with the reviewer over any needed changes until the reviewer accepts and merges your branch.
11. Delete your local copy of your branch.


### License
This project is Copyright (c) Johannes Sahlmann STScI/AURA and licensed under
the terms of the Aura license. This package is based upon
the `Astropy package template <https://github.com/astropy/package-template>`_
which is licensed under the BSD 3-clause licence. See the licenses folder for
more information.