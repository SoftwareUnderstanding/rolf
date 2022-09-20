# Compact Group Package

## High Level Overview
This repository contains the code for reproducing the results of Wiens et al 2018. 
The scripts will download the Millennium Simulation data, analyze the data for Compact Groups and then plot the results. 

## Usage Instructions
For best performance, it is recommended to have access to at least 1 TB of 
high bandwidth storage as well as a server with 20+ cores. 

The data should be downloaded before processing as it will take a considerable time to 
download and waste CPU resources if running on a shared cluster.

### Setup 
First, the user will need to register for an with access to the Millennium (http://gavo.mpa-garching.mpg.de/Millennium/Help?page=registration) 
database. Once proper credentials are received, then the data may be downloaded. 

To download the data, one must run the script `get_data.py` and allow for several hours 
as the data is downloaded. Unfortunately, there is no compression applied to the data from
the server. 

Example usage:

`python get_data.py --username my_username --password my_password`

Required Arguments:

* `--username` Username for Millennium Database Access
* `--password` Password for Millennium Database Access

Optional Arguments: 

* `--snapnums` List of snapnums to be downloaded
* `--size` Length of a side of the dividing boxes in Mpc 
* `--annular_radius` Overlap in Mpc between boxes. Should be equal to the annular radius value used in `main.py`
* `--outdir` Directory where data should be saved
* `--overwrite` Overwrites any existing data in the outdir if it exists


### Data Processing
After the data has been downloaded in reasonably sized chucks, it is ready to be processed. 
This script is at the heart of the project as it takes the raw simulation data, clusters it and 
performs analytics on each discovered group. 

If the user does not specify any arguments, the script will run the default analysis using the Mean-Shift algorithm for 
clustering. However, this is not recommend as the DBscan algorithm has far better performance.


Example Usage:
```
snaps=`seq -s ' ' 0 63`
python  main.py --use_dbscan --snapnums $snaps --num_cpus 20 --neighborhood 0.050 --max_annular_mass_ratio 1.e-4 --min_secondtwo_mass_ratio 0.10 --outdir results/default/ --datadir data/ --dwarf_limit 0.05
```

Arguments:

* `--snapnums` Space separated list of Snapnums to process
* `--size` Length of Simulation chunk (cube) side. Must match the `--size` option specified in `get_data.py`
* `--cluster` Forces the script to rerun the clustering algorithm even if cache results exist
* `--use_dbscan` Uses DB scan algorithm instead of Mean-Shift
* `--neighborhood` Radius for the DB Scan Algorithm (if `--use_dbscan` selected)
* `--bandwidth` Bandwidth parameter for the MeanShift Algorithm
* `--min_members` Minimum number of galaxies to be considered a Compact Group
* `--dwarf_limit` Minimum Stellar Mass (10<sup>10</sup> M<sub>sun</sub>) of galaxy to be included in a Compact Group
* `--crit_velocity` Maximum Velocity (km s<sup>-1</sup>) difference between a galaxy and the median group velocity
* `--annular_radius` Size of outer radius for annular mass ratio calculation
* `--max_annular_mass_ratio` The maximum ratio of mass in annulus to mass in compact group
* `--min_secondtwo_mass_ratio` Minimum ratio of mass of the second and third most massive galaxies to the most massive
* `--mass_resolution` Smallest galaxies to consider in the analysis (for trimming out galaxies without stellar mass)
* `--num_cpus` Number of CPUs to use in parallel
* `--profile` Ipython Profile for multinode computing. Default is to use multiprocessing for single node multicore processing
* `--outdir` Directory to save the results in 
* `--datadir` Directory containing the data
* `--overwrite` Overwrites any previous results in the output directory 
* `--verbose` Outputs Verbose Log Messages
* `--nolog` No log files will be saved
* `--test` Runs the program with only on chunk of the simulation instead of the entire thing


### Group Evolution
To perform the analysis on the history of galaxies in compact groups, the groups need to be traced through the entire 
simulation. The data structure is a Tree so finding the descendants of a galaxy involves tracing the branch down to the
root and saving the results. 

#### Setup 
First, the data needs to be loaded into a Postgres database. It is recommended to load this database onto a SSD or 
similar high speed storage so data retrieval does not become the bottle neck. The columns used in evolution tracing are 
`"galaxyid", "redshift", "firstprogenitorid", "descendantid", "treeid", "snapnum"`allowing the user to only load these 
into the database instead of all of the downloaded data. 

Recommended Steps:
1. Trim the data for the desired columns and save to a csv for each snapnum
2. Create the table in the DB 
3. Issue copy commands for each CSV file
4. Create an index on the TreeId column (This column is used most when querying the data)


#### Running the Evolution Script
It is recommended to run the script with 4+ cores as it is another extremely parallel workflow. Each core opens a connection
to the DB and iterates through its list of treeIds doing the tracing for each Compact Group galaxy. To save the results,
each core places the results into a queue which the writing thread reads from to ensure the writes are ACID. 

Example Usage:
`python evolution_tracing.py --pool_size 16 --group_file_location /path/to/group/file.csv`

Arguments:
* `--pool_size` Number of CPUs to use
* `--group_file_location` File contains all Compact Group galaxies to be traced


### Plotting the Results
The plotting script will take the results from the main analysis and the evolution script to create the plots 
found in Wiens et al 2018. These plots are generated using LaTex formatting and saved in EPS format. 

Example Usage
```
python plots.py --resultsdirs /data/compactgroups/tvw/results/nh_0.075_ar_1e-4_st_0.10/ /data/compactgroups/tvw/results/nh_0.050_ar_1e-4_st_0.10/ /data/compactgroups/tvw/results/nh_0.025_ar_1e-4_st_0.10/ \
                --datadir /data/compactgroups/tvw/data/ \
                --labels "{\sc nh}$=75$ $kpc$ $h^{-1}$" "{\sc nh}$=50$ $kpc$ $h^{-1}$" "{\sc nh}$=25\$ $kpc$ $h^{-1}$" \
                --plotlabel "nh" \
                --snapnum_file snapnum_redshift.csv
```

Arguments:
* `--resultsdirs` Directories which contain the outputs from main analysis scripts
* `--plotlabel` Label of the plot
* `--datadir` Data where raw snapnum data is stored (directory containing the downloaded data from `get_data.py`)
* `--plotsdir` Directory to save the plots
* `--labels` Labels for data plotted in LaTeX formatting
* `--counts_file` File containing the galaxy counts for each snapnum. Will be recomputed if it doesn't exist
* `--snapnum_file` File containing the conversions for snapnum value to redshift
* `--dwarf_limit` Stellar Mass Limit for dwarf galaxies (only used when computing counts file)
* `--evolution` Run the plots for the evolution (Default is galaxies at each snapnum)




