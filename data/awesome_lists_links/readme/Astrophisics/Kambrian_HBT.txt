The Hierarchical Bound-Tracing subhalo finder and merger tree builder, for numerical simulations in cosmology. HBT tracks haloes since their birth and continues to track them after mergers, finding self-bound structures as subhaloes and record their merger histories as merger trees.

Usage:

--If reading gadget simulation, and you have the FoF halos already output in gadget format as well:

1.Create an parameter file under param/ directory, with name paramMyRun.h. You can find several examples there.

2.set RunName=MyRun in Makefile.runs

3.make

4.modify HBTrun or HBTjob etc. to submit your job.

--If reading other simulation format:
create your own io directory
change IODIR to be your own io directory in Makefile.runs
then follow the steps above for gadget simulation.


The HBT paper is available at: 
http://arxiv.org/pdf/1103.2099

Currently, limited Documentation and examples can be found under utils/ and io_example/.

Contact:
Jiaxin Han
ICC, Durham University
hanjiaxin #at# gmail.com