![alt text](https://github.com/rhyspoulton/WhereWolf/blob/master/wherewolf_logo.png)
# WhereWolf

Developer: Rhys Poulton

This is a code to continue to track halos (or galaxies) after they have been lost by the halo finder. For the results of using this see Poulton et al., submitted. For further results and a detailed description of how the code works see Poulton et al. in prep.

## Running

To make the reading of the gadget files much more efficient a mapping of the particle ID's needs to be created. The code genPartSortIndexFiles.py generates these files, it can be run by the command:

	python genPartSortIndexFiles.py <Gadget folder input directory> <Output directory> <snapshot>

Where this will need to be run for each <snapshot> desired (many of these can be run in parallel).

Once they have been created for all of the snapshots desired, then 4 file-list need to be created all in snapshot order. These are

1. List of the base gadget filenames for each snapshot (without the extension or mpi file number)
2. List of the base VELOCIraptor filenames for each snapshot (without the .propeties or mpi file number)
3. List of TreeFrog files for each snapshot
4. List of the base filenames created in the command above (again without extension or mpi file number)

These files needs to be updated in the "wherewolf.cfg" file. Please put the full paths to these files.

WhereWolf is then run by the command:

	mpirun -np <# of cpus> python wherewolf.py -c wherewolf.cfg -n <numsnaps> -o <output directory> [-s <snapshot offset>]

Snapshot offset only needs to specfied if starting at a non-zero snapshot. This assumes that the filelist contains the list of all the snapshot files in the simulation.

Note that WhereWolf is currently heavily I/O bound so there will be little difference in the timings for lots of MPI threads.

## Output

WhereWolf output files differenly for the VELOCIraptor halo catalogue and the TreeFrog merger-tree.

### VELOCIraptor

WhereWolf will add halos to the VELOCIrator catalogue by each mpi thread adding on a additional mpi file and will update the information in the original VELOCIraptor files

### TreeFrog

For TreeFrog, WhereWolf will create a new tree file for each snapshot that halos have been ghosted for. These files will have a .WW.tree extension. A file containing the names of the tree per snapshot will be created in \<output directory>/treesnaplist.txt.

### WhereWolf run statistics

As a diagnostic check WhereWolf also creates a file containing the statistics of why halos terminate tracking per snapshot. This can be useful in comparing different halo catalogues. This file is created in \<output directory>/WWrunstat.txt.

## Notes

This code is still currently under development so please contact me if you run into any problems or you think something can be improved
