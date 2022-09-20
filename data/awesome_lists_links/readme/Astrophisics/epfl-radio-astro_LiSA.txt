# LiSA
LIghtweight Source finding Algorithms for analysis of HI spectral data. This code was developed for the SKA Science Data Challenge #2.

## Installation
This pipeline requires python 3. The required python libraries can be installed the `setup.sh` script:

```bash
source setup.sh
```

## Structure
Several different modules are available for data analysis in the `modules` directory. All algorithms interface through the data via the `domain_reader.py` classes, which keeps track of the domain decomposition. The pipeline developed for SDC2 is given in `pipelines/pipeline.py`, and uses all of the modules in this library. The structure of the pipeline is shown in the Figure below:
<p align="center">
  <img src="https://github.com/epfl-radio-astro/LiSA/blob/main/pipeline.png" width="450" title="full data processing pipeline">
</p>

## Running
To run the modules, adjust the config file to point to the location of your data and launch the pipeline as follows:


```bash
python pipelnes/pipeline.py path-to-config-file total-number-of-domains domain-number
```

We also provide example batch scripts in the `pipelines` directory which run all domains using sarray.
