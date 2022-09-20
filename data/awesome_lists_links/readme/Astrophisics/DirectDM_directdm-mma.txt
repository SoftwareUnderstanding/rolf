**DirectDM**: a tool for dark matter direct detection
=====

`DirectDM` takes the Wilson coefficients of relativistic operators that couple DM to the SM quarks, leptons, and gauge bosons and matches them onto a non-relativisitc Galilean invariant EFT in order to caclulate the direct detection scattering rates.

You can get the latest version of `DirectDM` on [`github`](https://directdm.github.io).

## Usage

The included `example.m` file has basic examples for using the functions provided by the code. 

Here is a simple example assuming that the notebook or `.m` file are in same directory as `./DirectDM/`.

Load the package
```
AppendTo[$Path,NotebookDirectory[]];
<<DirectDM`
```

Set the DM type to a Dirac fermion and set some Wilson coefficients in the 3 flavor basis
```
SetDMType["D"]
Do[SetCoeff["3Flavor", Q6[1,f], 1/100^2], {f,{"u","d"}}]
SetCoeff["3Flavor", Q7[1], 1/100^3]
```

Match the 3 flavor Wilson coefficients onto the non-relativistic ones
```
ComputeCoeffs["3Flavor","NR"]
```

Get the list of proton and neutron NR Wilson coefficients (separate lists) 
```
CoeffsList["NR_p"]
CoeffsList["NR_n"]
```

## Citation
If you use `DirectDM` please cite us! To get the `BibTeX` entries, click on: [inspirehep query](https://inspirehep.net/search?p=arxiv:1708.02678+or+arxiv:1707.06998+or+arxiv:1611.00368&of=hx) 



## Contributors

   * Fady Bishara (University of Oxford)
   * Joachim Brod (TU Dortmund)
   * Jure Zupan (University of Cincinnati)
   * Benjamin Grinstein (UC San Diego)

## License
`DirectDM` is distributed under the MIT license.


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
