# OrbWeaver

## Developed by
	Rhys Poulton
	Additional contributors:
	Lucie Bakels
	Pascal Elahi

OrbWeaver is designed to extract orbits from halo catalogs enabling a large statistical study of their orbital parameters. This code is presented in Poulton et al., in prep, along with some results.

## Compilation

Orbweaver uses cmake as a build tool as it does many system-level checks, like looking for libraries and setting up the rules for the build, and then generates the actual build scripts in one of the supported build systems. It also is able to download the required submodules of this directly which this repository is dependant on. The simplest way to build is to use the following steps from the root of the repository

	mkdir build
	cd build
	cmake ..
	make all


## Running

There are two stages to running OrbWeaver the first is to run to generate a preprocessed catalogue, which contain a superset of orbiting halo for each identified orbit host. The second stage is to then run OrbWeaver on the preprocessed data to create the orbit catalogue which is the final output.


### Generating the preprocessed catalogue

 The code to generate these preprocessed files are run by the command:

	python OrbitCatalogCreation/CreateOrbitCatalog.py -c <configuration filename (example_configuration.cfg)> -i <file containing a list of the base VELOCIraptor filenames> -t <walkable tree file> -o <output base filename>

In the configuration file, you can specify the input catalog format, this will load in the datasets that are specified from the following files:

| Input catalog name | Name of dataset names files |
|--------------------|-----------------------------|
| VELOCIraptor       | OrbitCatalogCreation/example\_inputs/input\_VELOCIraptor\_catalog.txt |

*If the desired input catalog is not specified here it is not currently supported, please contact the lead developer.*

where the datasets loaded can be changed if desired and additional datasets can be loaded (coming soon). You can also modify the orbit host selection and the region around orbit host which is used for the superset of orbiting halos, in the configuration file. Each orbit host has an orbit forest (that contains halos that ever passed within the region of interest), and the code outputs a file that contains multiple orbit forest within them, the number is set by numOrbitForestPerFile in the configuration file. So multiple preprocessed files are created with the naming scheme:

	<output base filename>.<fileno>.orbweaver.preprocessed.hdf

A file containing the list of filenames is also outputted with the naming scheme

	<output base filename>.orbweaver.filelist.txt

### Creation of the orbit catalogue

Using the filelist generated in the first stage it is now possible to to run OrbWeaver using the python script in python-tools from the following command:

	python python-tools/runorbweaver.py -i <output base filename>.orbweaver.filelist.txt  -s <schedulertype> [-f <fracrvircross> -c <iclean> -v <verbose>]

Where:

* schedulertype - The type of scheduler avalible either Slurm, PBS or None, if None then python's multiprocessing will be used to run orbweaver concurrently (currently not implemented)
* fracrvircross - is the fraction of the host viral radius where a crossing point is outputed (default is a output point every 0.5 of the host viral radius)
* iclean  -  Flag to switch on/ off orbit cleaning, this is done to remove any apsis points where the object is not orbiting the host of interest. (default True)
* verbose - How verbose the code is 0=None, 1=talkative

If either a scheduler is used then the base submit file will needed to be updated based on the size of the data being processed, these are located in:

* Slurm: python-tools/examples/runorbweaver.base.sbatch
* PBS: python-tools/examples/qsub.runorbweaver.base.sh

This will submit multiple jobs (or processes) where a job is run per preproccesed file generated in the first stage.

Once these have run the data will be output to a file naming scheme for the orbitdata is:

	<output base filename>.<fileno>.orbweaver.orbitdata.hdf

## Output

For a full description of the output datasets please see the FieldsDescriptions.md.

## Reading in the data

To read in the data please use the associated python tool located in python-tools/orbweaver_tools.py and use the function ReadOrbitData which uses the filelist generated in the preprocessed stage as a input.
