# create_SimSpinFile
A script to take simulation files and output a new HDF5 file for use with SimSpin (https://github.com/kateharborne/SimSpin).

Requires imports from [pynbody](https://github.com/pynbody/pynbody) and [h5py](https://www.h5py.org/).

```
>> python create_SimSpinFile.py -h

usage: create_SimSpinFile.py [-h] [-i INFILE] [-o OUTFILE]

optional arguments:
  -h, --help                     show this help message and exit
  -i INFILE, --infile INFILE     The path to the simulation file you wish to process with SimSpin
  -o OUTFILE, --outfile OUTFILE  The path to the SimSpin HDF5 file produced
  
```
