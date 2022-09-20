# Eclairs
Author: Ken Osato  
Contributors: Takahiro Nishimichi, Francis Bernardeau, and Atsushi Taruya  


## Prerequisite
C++ compiler (`g++`, `icpc`, or `clang++`)  
GNU Scientific Library (GSL)  
Boost (optional for python wrapper)

The code is tested with icpc v19.0.0.117, GSL v2.5, Boost v1.66, and python v2.7.  
For compling this code, please specify the paths for these libraries
in `Makefile`. Then, type

```
make
```

Then an executable `eclairs` is created. For running the code,
you can pass the initial parameter file as,

```
./eclairs [initial parameter file]
```

You can find an example parameter file as [example.ini](example.ini).
If you pass nothing, the code runs with default parameters.


## Tutorial for python wrapper
For tutorial, you can refer to [tutorial.ipynb](tutorial.ipynb).


## Future plans
Currently, the computation of real-space matter power spectrum
based on RegPT and SPT is supported.
Additional modules, which incorporate redshift space distortion,
galaxy bias, and fast calculation based on RegPT-fast schemes,
will be available after companion papers are released.


## License
This code can be distributed under MIT License.
For details, please see the LICENSE file.  
If you use this code in your work, please cite the following paper.

[Osato, Nishimichi, Bernardeau, and Taruya, ArXiv:1810.10104](https://arxiv.org/abs/1810.10104)
