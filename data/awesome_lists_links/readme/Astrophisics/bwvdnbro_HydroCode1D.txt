# HydroCode1D

Powerful 1D finite volume code that can run any problem with 1D or 2D/3D
spherical symmetry including external gravity or self-gravity.

To run the code, download a copy of this repository. Then, configure the code
using `cmake` in a folder of choice:
```
cmake -DCMAKE_BUILD_TYPE=Release PATH_TO_SOURCE
```
where `PATH_TO_SOURCE` is the path to the folder containing `CMakeLists.txt`.
We strongly recommend configuring the code in a different folder from
`PATH_TO_SOURCE` to avoid problems when configuring different versions. Example
configurations are provided in the bash scripts `configure_sod1d.sh`,
`configure_sod2d.sh`, `configure_sod3d.sh`, `configure_bondi.sh`,
`configure_blastwaves.sh` and `configure_evrard.sh` (and correspond to special
initial conditions. To help setting up a configuration, we also provide the
script `get_cmake_command.py`, which can also be used as a Python module.

Once configured, run `make` from the folder where `cmake` was run:
```
make
```
This will compile the program. Note that the program consists of only 3 files,
so trying to speed up the compilation (using `make -j 4` to e.g. build with 4
threads) will likely not speed it up significantly. Note also that the initial
conditions for the simulation need to be provided in the file `UserInput.hpp`.
If no predefined configuration is used, you need to implement this before the
compilation (otherwise the compilation will fail).

Once successfully compiled, the program can be run using
```
./HydroCode1D
```
from the same folder where `make` and `cmake` were run. The program will provide
ample output during the run, and will, depending on the configuration, create a
(large) number of output files named `snapshot_XXXX.txt`. These contain the
midpoint position, density, velocity and pressure for each cell in the grid (in
SI units). The program will by default use all available threads (as given by
the environment variable `OMP_NUM_THREADS`). This can be overwritten by giving
the desired number of threads as a command line argument to the program.
