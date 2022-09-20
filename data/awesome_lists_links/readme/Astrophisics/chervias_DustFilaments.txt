DustFilaments
=====

DustFilaments is a code for painting filaments in the Celestial Sphere, to generate a full sky map of the Thermal Dust emission at millimeter frequencies by integrating a population of 3D filaments.

Available maps
--------------

Our maps are located at [http://astrophysics.physics.fsu.edu/~chervias/](http://astrophysics.physics.fsu.edu/~chervias/).
The units of our maps are uK Thermo.
We create full sky maps of T,Q,U emission at 20, 27, 39, 93, 145, 225, 280 GHz. Also we include maps at 217 and 353 GHz. We include maps with the T filament map, and the Q,U large scale filled by the Planck template (labeled `Hybrid-PolNoGPlane`), as well as the unit-calibrated map without this large-scale filled and before filtering by a high-pass filter. This is the raw Q,U map directly from our filament code, but calibrated to uK thermo units (labeled `Filaments-Raw`).
This is the model described in our paper, whose main parameters are listed in Table 1.

The code
--------

The code consists of 3 main methods:
* **get_MagField**: this function will return a magnetic field cube. The random isotropic component can be made deterministic by using the same seed again. Optionally you can provide a precalculated cube for e.g. the large scale Galactic component. This must be as a numpy array saved as a npz file, with a key label `Hcube`. Also note that the convention for the entire code is that a magnetic field cube has shape `[Npix,Npix,Npix,3]` and the indices ordering is `[index_z,index_y,index_x,:]`. You can also skip this `get_MagField` function all together if you want, if you want to calculate a magnetic field cube on your own, provided that these convention are followed. 

* **get_FilPop**: this function will create a filament population and has the magnetic field cube as input. It will create a random population with a given seed. Remember to always use the same magnetic field and population seed if you want to make two or more runs of the code if you are running too many frequency channels. 

* **Paint_Filament** this is the main method of the code, and it will paint a single filament into a healpix map provided as input. The healpix map is updated in place. The `test` directory has an example of a script that will run the code in a cluster using mpi4py with mpiexec.

Install
-------

**Requirements** : Standard python modules
* numpy
* healpy
* mpi4py
* yaml

Also, the healpix c++ library is needed for compiling the filament paint code.
To install, run 
```
python setup.py install --user
```
After install, you should also install [this](https://github.com/huffenberger-cosmology/magnetic-field-power-spectrum) code that generates the isotropic random magnetic field box. This code is needed if you use the *get_MagField* function.

Using the code
--------------

We provide a script that can be run in a cluster with mpi. The parameters must be specified in the file `test/params.yml`. To run, use
```
mpiexec -n Np python test/Example-script.py test/params.yml
```
where `Np` is the number of processes you want to run in your cluster. Each process will run `Nfils/Np` filaments, where `Nfils` is the total number of filaments you want to produce. One of the parameters is `Nthreads` which sets how many threads to run each call of the filament painting method. For a node of 64 threads, a good combination is running 8 mpi processes with 8 threads each, for example. Using a `hostfile` in mpi, you can easily run in multiple nodes to speed up the calculation.

**A note about memory use** This python script can run simultaneous frequency bands and produce simultaneous frequency maps. However, each mpi process will need to keep in memory a TQU map at your chosen Nside resolution plus a TQU map at every resolution below until Nside=128. You also need to multiply by the number of frequencies since this is for every frequency channel. Also, each mpi process needs to keep the 3D cube of the magnetic field in memory. The point is that memory builds up fast, so please consider this when trying to run too many frequencies at the same time. If repeating the seed for the magnetic field and for the filament population, you can split your run.
