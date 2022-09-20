# Indri - the pulsar population synthesis toolset
*by* Marek Cieślar, Tomasz Bulik, Stefan Osłowski

This program was made as a tool to study the population of single (not in binary or hierarchical systems) Neutron Stars.
Given a starting distribution of parameters (birth place, velocity, magnetic field, and period), the code moves a set of stars through the time (by evolving spin period and magnetic field) and the space (by propagating through the Galactic potential). Upon completion of the evolution, a set of observables is computed (radio flux, position, dispersion measure) and compared with a radio survey such as the Parkes Multibeam Survey. The models' parameters are optimised by using the Markov Chain Monte Carlo technique. 

The project was effectively started in the January, 2014 and ended in the January, 2018. Through this period the development was financially supported form the following sources:

* Einstein Telescope Consortium by Tomasz Bulik,
* Mistrz2013 subsidy by Krzysztof Belczyński,
* Astronomical Observatory, Warsaw University - mini research-grants for young scientists.

## Indri
<a title="By Zigomar (Own work) [GFDL (http://www.gnu.org/copyleft/fdl.html) or CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0)], via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File%3AIndri_Head.jpg"><img width="256" alt="Indri Head" src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/Indri_Head.jpg/512px-Indri_Head.jpg"/></a>

The project was named after critically endangered species of lemurs - Indri. Should you find the code useful, please consider donating to [World Wildlife Fund](https://support.worldwildlife.org/site/SPageServer?pagename=main_onetime). Thank you!


## Citing our work

If you use our work in your publication or as a jump-start for your own research, please cite our work ([MNRAS](https://dx.doi.org/10.1093/mnras/staa073), [arxiv](http://arxiv.org/abs/1803.02397)):
```
@article{10.1093/mnras/staa073,
    author = {Cieślar, Marek and Bulik, Tomasz and Osłowski, Stefan},
    title = "{Markov Chain Monte Carlo population synthesis of single radio pulsars in the Galaxy}",
    journal = {Monthly Notices of the Royal Astronomical Society},
    volume = {492},
    number = {3},
    pages = {4043-4057},
    year = {2020},
    month = {01},
    issn = {0035-8711},
    doi = {10.1093/mnras/staa073},
    url = {https://doi.org/10.1093/mnras/staa073},
    eprint = {https://academic.oup.com/mnras/article-pdf/492/3/4043/32215211/staa073.pdf},
}


```

In case of forking the code - please, make sure to cite the external sources used therein as well!

## Requirements

The majority of the code is written in the `C++` language. However, the integrated free electron model (NE2001) does require Fortran compliter. To successfully compile our work following packages/libraries should be installed:

- cmake >= 2.8
- g++ with c++0x implementation 
- gfortran
- hdf5 >= 1.8.6

To install the required packages on a Ubuntu system issue following commands:
```
apt-get update
apt-get install gcc g++ gfortran cmake libhdf5-dev
```

## External libraries/subroutines
A number of external libraries were used to create the simulation program. Those include:
- [MersenneTwister](https://doi.org/10.1145/272991.272995)
- [NE2001](http://www.astro.cornell.edu/~cordes/NE2001/) (the primary subroutines for the DM computations)
- [pugixml](https://pugixml.org)
- [PSREvolve](https://astronomy.swin.edu.au/~fdonea/psrevolve.html)
- some snippets (marked inside the code)
- [hdf5](https://support.hdfgroup.org/HDF5/)
- [healpix](https://healpix.jpl.nasa.gov/) (used to develop some ideas that are not production level ready)

## License 

Please note that content is subject to license and copyright by respective content creators and entities (listed above). The code created by us is licensed under the GNU General Public License v3.0.


## Installation

After obtaining the source:
```
git clone 
cd Indri
```

Make the `build` directory and issue the cmake inside it:
``` 
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
```

Then, to build the project:
```
make
```

Compiled programs will be located at _build/bin_ sub-directory.

Due to the fact that cmake makes directories (_build/bin/input_ and _build/bin/Scripts_) and populates them by a means of *not so kosher hacking*, before issuing another `cmake ..` please do the `rm -fr *` inside the _build_ directory. Otherwise the cmake will complain a lot. 

A copy of ATNF catalogue is needed to run anything. A necessary file should copy itself upon cmaking. To update the catalogue:
```
cd build/bin/Scripts
./GetCalalogue.sh
```
If it's not working, the ATNF probably changed versions (the `version=1.5` part in the address inside wget has to be corrected).



## Development
For a development build with clang goodies:
```bash
cd build
CXX='cc_args.py g++' CC='cc_args.py gcc' cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_BUILD_TYPE=Debug ..
make
find . | ag clang_complete | xargs cat | sort | uniq > ../.clang_complete
```
### State of the code
This project was developed on short-time scales (typically on one/two-month goals). 
We deem the results stable, however the code itself may be well beyond the point 
of easy maintenance as the final state does not reflect any plan or design. 
The view of the PI (MC) is that it would be easier to reuse 
only parts of the project as a jump-start for the further development. 

The code contains a lot of experimental features. Some are tested and some are only
prototyped. It is advised to assume that everything outside the main computational
path could produce incorrect results.

### Making doxygen documentation
Inside the directory `Docs` there is attached file with doxygen configuration. To generate html documentation of the code issue:
```
cd Docs
doxygen Doxyfile.config
```
The output *index.html* will be placed in _Docs/html/_ folder.
The configuration uses relative paths and should run with any placement of the project directory.
The docs contain both *hierarchy* and *call* graphs. 

## Reproducing the results
The reproduction of the results presented in our paper could be done with a small cluster (~250 processors) in about a week or two.
Whole process id divided into three parts:

1. Building reference population
2. MCMC simulations

Note that executing most (if not all) of the programs without any argument or with `-h` will print a help message with arguments usage.
Also, the `input` subdirectory (autmatically copied by the `cmake`) must be present in the path were programs are executed. It may be esier to `cd` to the `bin` directory and run stuff from there.

### (1) Building the reference population
The purpose of 1 is to cut the time due to the DM integration which takes considerate amount of time. Simply put - a large amount of Neutron stars are simulated and then we reuse their positions while changing their P-Pdot-B evolution and emission properties (see our paper for details).

To simulate the reference population run the `PopulationMaker`:
```
./PopulationMaker -o PopRaw.1.bin -n 1000000
```
which will simulate 1M neutron stars and write the output to `PopRef.1.bin`


After the simulation it is imperative to sub-select the reference population for only those neutron stars that can actually make into the Parkes Survey field of view. To cut the results run the `PopMeter`:
```
./PopMeter -C -i PopRaw.1.bin -o PopRef.1.bin
```
This subroutine will also disregard pulsars that are were thrown out of the Galaxy (distance from the centre is grater than 35[kpc]).

Note that the efficiency of the geometric model is 16% - meaning that 84% of pulsars is lost. Since a *pulsar* in written as a structure, one can join multiple binary files by simply attaching them an output `PopRef.bin`:

```
cat PopRef.1.bin >> PopRef.bin
```

Now we can create reference population with specific size:
```
#100k
./PopMeter -C -i PopRaw.bin -o PopRef100k.bin -NOut 100000
#1M
./PopMeter -C -i PopRaw.bin -o PopRef1M.bin -NOut 1000000
#10M
./PopMeter -C -i PopRaw.bin -o PopRef10M.bin -NOut 10000000
```
Due to our computational capabilities we use 1M files. 

#### Newer free electron model
The default DM computations are done by using the canonical [NE2001](http://www.astro.cornell.edu/~cordes/NE2001/) free electron model. To utilize the newer [YMW16](www.xao.ac.cn/ymw16/) model use the `DMChanger4YMW16` executable:

To extract the DM data:
```
./DMChanger4YMW16 -prep -i PopRef.bin -odm DMNE2001.txt
```
The file contains:
```
NR GalacticL[deg] GalacticB[deg] Distance[kpc] DM[pc/cm^3]
```

Obtain and compile the [YMW16](www.xao.ac.cn/ymw16/) code (for our research we used v1.2.2 of the model). Then one can change the DM values by using a script:
```bash
#!/bin/bash

if [ -d /dev/shm/ymw16.temp ]
then
        rm -fr /dev/shm/ymw16.temp
fi
mkdir /dev/shm/ymw16.temp
cp DMNE2001.txt spiral.txt ymw16par.txt /dev/shm/ymw16.temp

while IFS='' read -r line || [[ -n "$line" ]]; do

#GL=`echo $line | awk '{print $2}'`
#GB=`echo $line | awk '{print $3}'`
#DIST=`echo $line | awk '{print $4*1000}'`
INPUT_PAR=`echo $line | awk '{print $2, $3, $4*1000}'`
RES=`./ymw16 -d /dev/shm/ymw16.temp/  Gal $INPUT_PAR 2 |  awk '{print $9, $11}'`
#DMaTAU=`echo $RES | awk '{print $9, $11}'`
#DM=`echo $RES | awk '{print $9}'`
#echo "$line $DM" | awk '{print $1, $2, $3, $4, $6}'
echo "$line $RES" >> /dev/shm/ymw16.temp/DMYMW16.txt

done < "/dev/shm/ymw16.temp/DMNE2001.txt"

mv /dev/shm/ymw16.temp/DMYMW16.txt .
```

To import back the updated DM values:

```
./DMChanger4YMW16 -change -i PopRef.bin -idm DMYMW16.txt -o PopRef.YMW16.bin
```

The exact file with the reference population used in the computation is located [here](http://sirius.astrouw.edu.pl/~mcie/antares/IndriP1/PopRef.1M.50Myr.YMW16.bin) (md5sum: 234c6ea3a6b685e732b3118f0b347ac7)


### (2) MCMC

To run the MCMC computations isse:
```
./MCMCMaker -i PopRef.bin -o MCMC.All.1.bin -c MCMC.All.xml
```

To make a xml config file run `./MCMCMaker -h` command which will produce a `Example.MCMCConfig.xml` file. The config files used for the MCMC computations presented in the paper are located here: 

* for the *rotational* model - [MCMC.ERotAll.xml](http://sirius.astrouw.edu.pl/~mcie/antares/IndriP1/MCMC.ERotAll.xml) (md5sum: b58bda80608e9d1922b1596e60f8d28d)
* for the *power-law* model - [MCMC.All.xml](http://sirius.astrouw.edu.pl/~mcie/antares/IndriP1/MCMC.All.xml) (md5sum: 71dc9b99dff900fb41c43307e50c7df8)

Due to that some scripts in the pipeline may be name-sensitive some errors may arise
should the names be changed. Also note that some of variables are used in the log scale 
(fLPowerLawGamma and fBDecayDelta) despite the fact that they are written in the 
linear scale in the config file.

Assuming that a several of MCMC computations were completed, one can join them together
using the `MCMCChainCombiner`:
```
./MCMCChainCombiner MCMC.All.bin MCMC.All.1.bin MCMC.All.2.bin MCMC.All.3.bin
```

To change the internal, binary format to more useful `hdf` file:
```
./MCMCReadWrite  MCMC.All.bin  MCMC.All.hdf
```

