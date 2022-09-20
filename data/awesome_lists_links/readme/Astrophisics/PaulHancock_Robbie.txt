# Robbie: A batch processing work-flow for the detection of radio transients and variables

## Description

Robbie automates the process of cataloguing sources, finding variables, and identifying transients.

The workflow is described in [Hancock et al. 2018](https://ui.adsabs.harvard.edu/abs/2019A%26C....27...23H/abstract) and carries out the following steps:
- Preprocessing:
  - Find sources in images
  - Compare these catalogues to a reference catalogue
  - Use the offsets to model image based distortions
  - Make warped/corrected images
- Persistent source catalogue creation:
  - Stack the warped images into a cube and form a mean image
  - Source find on the mean image to make a master catalogue
  - Priorized fit this catalogue into each of the individual images
  - Join the catalogues into a single table and calculate variability stats
- Transient candidate identification:
  - Use the persistent source to mask known sources from the individual images
  - Source find on the masked images to look for transients
  - Combine transients tables into a single catalogue, identifying the epoch of each detection

## Dependencies
Robbie relies on the following software:
- [AegeanTools](https://github.com/PaulHancock/Aegean)
- [fits_warp](https://github.com/nhurleywalker/fits_warp)
- [Stils/TOPCAT](http://www.star.bris.ac.uk/~mbt/topcat/)

The best way to use Robbie is via a docker container which has all the sofware dependencies installed. Such a container can be built using `docker/Dockerfile`, or by pulling the latest build from [DockerHub](https://hub.docker.com/r/paulhancock/robbie-next) via `docker pull paulhancock/robbie-next`.

Robbie scripts are written for Python3. If you require Python2 compatibility then you should invest in a time machine.

## Quickstart
Robbie now uses Nextflow to manage the workflow. The `Makefile` is obsolete and no longer supported so don't use it.

Robbie can be run on a local system or on an HPC, and can use a container via singularity or docker, or can use software installed on the host. The current development cycle tests Robbie by using singularity on an HPC with the Slurm executor - other setups *should* work, but haven't been extensively tested.

### `main.nf` 
This file describes the workflow and can be inspected but shouldn't need to be edited directly.


### `nextflow.config`
This file contains all the configuration setup with default values. Copy this file and change these values to suit your data.

### Running

Robbie can be run via: `nextflow -C my.config run main.nf -profile common,Zeus -resume`

The `-C my.config` directs nextflow to use *only* the configuration described in `my.config`. If you use `-c` then it will also read the `nextflow.config` file. The `-profile common,Zeus` allows you to use the `common` profile settings that are described in the config file, and `Zeus` uses the settings specific for the Zues cluster on Pawsey.

Additional configuration files are stored in the `./config` directory, and may be useful templates for your work.

## Credit
If you make use of Robbie as part of your work please cite [Hancock et al. 2018](http://adsabs.harvard.edu/abs/2019A%26C....27...23H), and link to this repository.

## Links
You can obtain a docker image with the Robbie dependencies installed at [DockerHub](https://hub.docker.com/r/paulhancock/robbie-next/)
