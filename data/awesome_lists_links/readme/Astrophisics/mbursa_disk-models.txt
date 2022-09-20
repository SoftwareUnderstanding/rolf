# Accretion Disk Radial Structure Models

A collection of radial structure models of various accretion disk solutions, i.e. the radial dependence of their geometrical, physical and thermodynamic quantities.

## Models

The following models are included:

* [Novikov-Thorne thin disk model](models/nt) (in models/nt)
* [Sadowski polytropic slim disk model](models/slimdisk) (in models/slimdisk)

See the model's README files for more information and usage instructions related to each model.

## Installation

To download and compile the model collection use
```bash
git clone --recursive https://github.com/mbursa/disk-models.git
cd disk-models
make
```

The recursive option makes sure all submodules are included too (see bellow). 

Individual models can be compiled separately by running `make` in the directory of a particular model or by running
```bash    
make -C models/<model_name>
```    

### Dependencies
Some of the models may have external dependencies, which are handled using [git submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules) that are located under `libs` subfolder. Submodules are downloaded and initialized automatically when `git clone --recurse-submodules` (for version >=2.13 of git) or `git clone --recursive`  (for git versions >1.6.5 and <2.13) is used to clone the repository. 

For already cloned repos, or older git versions (<1.6), use the following to initialize submodules:
```bash
git clone https://github.com/mbursa/disk-models.git
cd disk-models
git submodule update --init --recursive
``` 

Most models use [SIM5](https://github.com/mbursa/sim5) library.

## Usage

The compilation produces a linux shared library (`.so` extension) for each disk model. These libraries can be linked to programs at compile-time or they can be loaded by a program dynamically at run-time. 

The models can be used quite independently. However, the idea has been that the models are used along with [SIM5](https://github.com/mbursa/sim5) library to compute and compare the observed spectra and other characteristics.

### Dynamic linking

<!--
```
gcc -Ldisk-models/nt main.c -l:nt -o your_program`
```
-->
tbd

### Loading at runtime
Run-time loading in a program can be done using [`dlopen()`](http://man7.org/linux/man-pages/man3/dlopen.3.html) function. 
```C
void* lib_handle;
int (*diskmodel_init)(double M, double a, char* params);
// open library
void* lib_handle = dlopen(modellib, RTLD_NOW);
// assign library function to a local variable
*(void**)(&diskmodel_init) = dlsym(handle, "diskmodel_init");
// execute library function 
diskmodel_init(10.0, 0.0, "");
// close the library
dlclose(handle);
```

A complete example of how to dynamicly load and use a disk model library in a program is provided in [examples/runtime-loading](examples/runtime-loading).


### Using in Python

The compiled shared object libraries with disk models can be also used in Python. In the root folder, there is a `disk_model.py` wrapper that handles loading the dynamic library and makes an interface to the library methods. A basic pattern of calling disk models in Python using `disk_model.py` is

```Python
root_path = 'path/to/disk_model.py'
sys.path.append(root_path)
from disk_model import DiskModel
model = DiskModel(root_path+'/models/nt/disk-nt.so', 10.0, 0.0, 'mdot=0.1')
```

A complete example of how to call disk model methods in Python is provided in [examples/python](examples/python).



### Loading with SIM5 library

[SIM5](https://github.com/mbursa/sim5) library contains a ready-to-use [interface for loading the disk models](https://github.com/mbursa/sim5/blob/public/doc/sim5lib-doc.md#sim5disk). It essentially uses  the approach described above and provides a C interface to access the functions of a disk model library that takes care of implementing the dynamic linking. A brief example how it can be used:
```C
#include "sim5lib.h"
// link the disk model libray disk-nt.so
diskmodel_init("./disk-nt.so", 10.0, 0.0, "mdot=0.1,alpha=0.1");
// call a funtion from the library (effective flux at raduis r = 10 r_g)
float F = diskmodel_flux(10.0);
// close the library
diskmodel_done();

```

## Interface

Each disk model implements a unified set of functions that specify the quantities that each model provides.

A complete description of the interface is given in [doc/interface.md](doc/interface.md).


## Citing

If you have used this software in your work, please acknowledge it by citing its ASCL record 
[ascl:2002.022](https://ascl.net/2002.022). The BibTeX citation record can be obtained from 
[ADS](https://ui.adsabs.harvard.edu/abs/2020ascl.soft02022B/exportcitation). 
See ASCL [citing guidelines](https://ascl.net/wordpress/about-ascl/citing-ascl-code-entries/).

In addition, you should also reference the papers related to the model you use. See the README files of the individual models.

## Contributing

I will be happy to include more models if I am given a C implementation of the quantities that are part of the interface.

## License

The code is released under the MIT License.

