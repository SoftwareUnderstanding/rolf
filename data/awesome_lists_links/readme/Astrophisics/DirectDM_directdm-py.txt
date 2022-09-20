**DirectDM**: a python package for dark matter direct detection
=====

`directdm` takes the Wilson coefficients of relativistic operators that couple DM to the SM quarks, leptons, and gauge bosons and matches them onto a non-relativistic Galilean invariant EFT in order to calculate the direct detection scattering rates.

You can get the latest version of `directdm` on [`github`](https://directdm.github.io).

## Installation

`directdm` needs `python3`, and the `python3` versions of `SciPy` and `NumPy`.

The latest release is available via the [Python Package Index](https://pypi.org/), so you can install it simply by executing

```
pip3 install directdm
```

(You might need root permissions to execute this command, and `pip3` might be called `pip` on your system.)

**Note**: the package has been tested on a linux machine. Mac users using `homebrew` should be able to install `directdm` after executing

```
brew install python3
brew upgrade numpy
brew upgrade scipy
```

## Usage

Here is a simple example how to use `directdm`:

Import the package:
```
import directdm as ddm
```

Define some Wilson coefficients for Dirac DM in the three-flavor basis, using a `python` dictionary:
```
wc_dict = {'C61u' : 1./100**2 , 'C61d' : 1./100**2}
wc3f = ddm.WC_3f(wc_dict, DM_type="D")
```

Match the three-flavor Wilson coefficients onto the non-relativstic ones (the DM mass has been set to 100 GeV, and the momentum transfer is 50 MeV):
```
print(wc3f.cNR(100, 50e-3))
```

Write the list of proton and neutron NR Wilson coefficients into a file in the current directory with filename 'wc3.m':
```
wc3f.write_mma(100, 50e-3, filename='wc3.m')
```

The included `USAGE.md` file has basic descriptions of all relevant classes.

The included `example.py` file has basic examples for using the functions provided by the code.


## Citation
If you use `DirectDM` please cite us! To get the `BibTeX` entries, click on: [inspirehep query](https://inspirehep.net/search?p=arxiv:1809.03506+or+arxiv:1801.04240+or+arxiv:1710.10218+or+arxiv:1708.02678+or+arxiv:1707.06998+or+arxiv:1611.00368&of=hx) 


## Main Author 

   * Joachim Brod (University of Cincinnati)


## Contributors

   * Fady Bishara (University of Oxford)
   * Benjamin Grinstein (UC San Diego)
   * Emmanuel Stamou (University of Chicago)
   * Jure Zupan (University of Cincinnati)


## License
`DirectDM` is distributed under the MIT license.


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
